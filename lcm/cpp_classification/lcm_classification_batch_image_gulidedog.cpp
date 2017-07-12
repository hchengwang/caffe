#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <signal.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>

#include <lcm/lcm.h>
#include <bot_core/bot_core.h>
#include <cv_bridge_lcm/rv-cv-bridge-lcm.h>
#include <lcmtypes/bot_core_image_t.h>
#include <bot_lcmgl_client/lcmgl.h>
#include <GL/gl.h>
#include "lcmtypes/april_tags_class_t.h"
#include "lcmtypes/april_tags_gd_class_t.h"
#include "lcmtypes/april_tags_gd_class_array_t.h"
#include <jpeg-utils/jpeg-utils.h>
#include <jpeg-utils/jpeg-utils-ijg.h>

#include <time.h>
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using std::string;

#define CPU_ONLY
int64_t start_time;
unsigned long GetTickCount()
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);

}

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	Classifier(const string& model_file,
			const string& trained_file,
			const string& mean_file,
			const string& label_file,
			const string& gpu_cpu);

    std::vector< vector<Prediction> > ClassifyBatch(const vector< cv::Mat > imgs, int N = 5);
	void set_lcm(lcm_t* lcm, std::string image_channel);
	void set_cpu(int is_cpu);
	void set_threshold(double pred_threshold, double diff_threshold);
	void set_batch_size(int batch_size);
	void set_output_number(int output_number);
	void set_motion_visual(int moition_visual);
	void run();
	void finish();

	// Message Handler
	void on_libbot(const bot_core_image_t *msg);
	static void on_libbot_aux(const lcm_recv_buf_t* rbuf,
			const char* channel,
			const bot_core_image_t* msg,
			void* user_data);

	void image_preprocess();
	/*static void on_quad_proposals_aux(const lcm_recv_buf_t* rbuf,
			const char* channel,
			const april_tags_quad_proposals_t* msg,
			void* user_data);*/
	void carcmd_visualization(april_tags_gd_class_array_t* gd_class_array);
    std::pair<float , float> tf_probs2twist(april_tags_gd_class_t gd_array);
    std::pair<float , float> tf_probs2vel(april_tags_gd_class_t gd_class);
	float tf_probs2omega(float prob);
	void draw_arrowimage(float tmp_v, float tmp_omega, int j, cv::Scalar color);

private:
	void SetMean(const string& mean_file);

	std::vector<float> PredictBatch(const vector< cv::Mat > imgs);
	
	void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);
	
	void PreprocessBatch(const vector<cv::Mat> imgs,
            std::vector< std::vector<cv::Mat> >* input_batch);

private:
	shared_ptr<Net<float> > net_;
	int is_cpu_;

	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;

	lcm_t* lcm_;
	cvBridgeLCM * cv_bridge_lcm_;
	// bot_lcmgl_t* lcmgl_;
	bool finish_;
	// threshold 
	double pred_threshold_;
	double diff_threshold_;
	// batch size
	int batch_size_;
	// image;
	cv::Mat img_;
	cv::Mat img_rgb_;
	std::vector<cv::Mat> imgs_;
	// get image;
	int image_number_;
	// output channel;
	int output_number_;
	// LCM Message Channels
	std::string image_channel_;
	int64_t start_time;
	bot_core_image_t_subscription_t* sub;
	// motion visualization
	bool motion_visual_;

};

Classifier::Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file,
		const string& gpu_cpu) {

	//if(this->is_cpu_ > 0){
	//	Caffe::set_mode(Caffe::CPU);
	//}else{
	//	Caffe::set_mode(Caffe::GPU);
	//}

	if(std::strcmp(gpu_cpu.c_str(),"CPU") == 0){
		Caffe::set_mode(Caffe::CPU);
	}else{
		Caffe::set_mode(Caffe::GPU);
	}

	/* Set threshold */
	//this->pred_threshold_ = 0.9;
	//this->diff_threshold_ = 3;

	/* Set batchsize */
	this->batch_size_ = 5;
	this->image_number_ = 0;
	this->motion_visual_ = 1;
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
	<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	
	
	/* Load the binaryproto mean file. */
	//;if(boost::filesystem::exists(mean_file)){
	//	SetMean(mean_file);
	//}

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
	<< "Number of labels is different from the output layer dimension.";
}

void Classifier::set_lcm(lcm_t* lcm, std::string image_channel){

	this->lcm_ = lcm;
	this->image_channel_ = image_channel;

	//this->lcmgl_ = bot_lcmgl_init(lcm, "LCMGL_MOCAP");
        //brian
        cv::Mat image;
        
	sub = bot_core_image_t_subscribe(
			lcm, image_channel.c_str(), Classifier::on_libbot_aux, this);

	cv_bridge_lcm_ = new cvBridgeLCM(lcm, lcm);
	//caffe_class_ = new april_tags_caffe_class_t;
	finish_ = false;

}

void Classifier::set_cpu(int is_cpu){
	this->is_cpu_ = 1;
}
void Classifier::set_threshold(double pred_threshold, double diff_threshold){
	this->pred_threshold_ = pred_threshold;
	this->diff_threshold_ = diff_threshold;
}
void Classifier::set_batch_size(int batch_size){
	this->batch_size_ = batch_size;
}
void Classifier::set_output_number(int output_number){
	this->output_number_ = output_number;
}
void Classifier::set_motion_visual(int motion_visual){
	if(motion_visual == 1){
		this->motion_visual_ = true;
	}else{
		this->motion_visual_ = false;
	}
}
void Classifier::run() {
	while(0 == lcm_handle(lcm_) && !finish_) ;
}

void Classifier::finish() {
	this->finish_ = true;
}

static bool PairCompare(const std::pair<float, int>& lhs,
		const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector< vector<Prediction> > Classifier::ClassifyBatch(const vector< cv::Mat > imgs, int num_classes){
    std::vector<float> output_batch = PredictBatch(imgs);
    std::vector< std::vector<Prediction> > predictions;
    for(int j = 0; j < imgs.size(); j++){
        std::vector<float> output(output_batch.begin() + j*num_classes, output_batch.begin() + (j+1)*num_classes);
        std::vector<int> maxN = Argmax(output, num_classes);
        std::vector<Prediction> prediction_single;
        for (int i = 0; i < num_classes; ++i) {
          int idx = maxN[i];
          prediction_single.push_back(std::make_pair(labels_[idx], output[idx]));
        }
        predictions.push_back(std::vector<Prediction>(prediction_single));
    }
    return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
	<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	 * filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector< float >  Classifier::PredictBatch(const vector< cv::Mat > imgs) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  input_layer->Reshape(this->batch_size_, num_channels_,
                       input_geometry_.height,
                       input_geometry_.width);

  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector< std::vector<cv::Mat> > input_batch;
  WrapBatchInputLayer(&input_batch);
  PreprocessBatch(imgs, &input_batch);
  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels()*imgs.size();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */

void Classifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch){
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i){
          cv::Mat channel(height, width, CV_32FC1, input_data);
          input_channels.push_back(channel);
          input_data += width * height;
        }
        input_batch -> push_back(vector<cv::Mat>(input_channels));
    }
    cv::imshow("bla", input_batch->at(1).at(0));
    cv::waitKey(1);
}

void Classifier::PreprocessBatch(const vector<cv::Mat> imgs,
                                      std::vector< std::vector<cv::Mat> >* input_batch){
    for (int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));
        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1)
          cv::cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
          cv::cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
          cv::cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
          cv::cvtColor(img, sample, CV_GRAY2BGR);
        else
          sample = img;

        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
          cv::resize(sample, sample_resized, input_geometry_);
        else
          sample_resized = sample;
      	//std::cout << sample_resized << std::endl;
        cv::Mat sample_float;
        if (num_channels_ == 3)
          sample_resized.convertTo(sample_float, CV_32FC3);
        else
          sample_resized.convertTo(sample_float, CV_32FC1);
        //cv::Mat sample_normalized;
        //cv::subtract(sample_float, mean_, sample_normalized);

      	cv::normalize(sample_float, sample_float, 0, 255, NORM_MINMAX, CV_32FC1);
      	//std::cout << sample_float << std::endl;
        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        //cv::split(sample_normalized, *input_channels);
        cv::split(sample_float, *input_channels);
        //CHECK(reinterpret_cast<float*>(input_channels->at(i).data)
        //      == net_->input_blobs()[0]->cpu_data())
        //  << "Input channels are not wrapping the input layer of the network.";
    }
}

void Classifier::on_libbot(const bot_core_image_t* image_msg) {

	cv::Mat im_rgb = cv::Mat::zeros(image_msg->height, image_msg->width, CV_8UC3);
	cv::Mat im_vis = cv::Mat::zeros(image_msg->height, image_msg->width, CV_8UC3);
	
	if(image_msg->pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG){
//		// jpeg decompress
		jpegijg_decompress_8u_rgb (image_msg->data, image_msg->size,im_rgb.data, image_msg->width, image_msg->height, image_msg->width * 3);
	}else {
		im_rgb.data = image_msg->data;
	}
	//im_rgb.data = image_msg->data;
//        //brian
//        this->image = im_rgb;
//        //
	cv::Mat im_rect = cv::Mat::zeros(im_rgb.rows, im_rgb.cols, CV_8UC1);
	cv::cvtColor(im_rgb, im_rect, CV_RGB2GRAY);
	// TODO: rectify image
	cv::Mat img;
	//cv::Mat img = cv::imread(file, -1);
	img = im_rect;
	CHECK(!img.empty()) << "Unable to decode image ";
	this->img_rgb_ = im_rgb;
	this->img_ = img;
	this->imgs_.push_back(im_rgb);
	if(this->imgs_.size() == this->batch_size_){
		this->image_preprocess();
		this->imgs_.clear();
	}
	
}

void Classifier::image_preprocess( ) {
	 // measure process time
	start_time = bot_timestamp_now();
	
    std::vector< std::vector<Prediction> > predictions;

    april_tags_class_t types[this->batch_size_][this->output_number_];
    april_tags_gd_class_t gd_array[this->batch_size_];
    april_tags_gd_class_array_t gd_class_array;
    
    predictions = this->ClassifyBatch(this->imgs_, this->output_number_);
    
	for (int j=0; j < this->batch_size_; j++)
	{
		for (int k = 0; k < this->output_number_; k++)
		{
			Prediction p = predictions[j][k];
			std::cout << p.second << " - " << p.first << std::endl;

            types[j][k].type = (char*)p.first.c_str();
            types[j][k].prob = predictions[j][k].second;                 
		}	
		std::cout << "--------------------------" << std::endl;
		gd_array[j].output_number = this->output_number_;
		gd_array[j].preds = types[j];
	}
    gd_class_array.gd_array = gd_array;
	gd_class_array.batch_size = this->batch_size_;
	april_tags_gd_class_array_t_publish(lcm_, "gd_class_array", &gd_class_array);
	if(this->motion_visual_){
    	this->carcmd_visualization(&gd_class_array);

    	std::stringstream drawn_image_topic;
    	drawn_image_topic << "image_with_motion_arrow";
    	for (int i = 0; i < this->batch_size_; i++){
     		cv_bridge_lcm_->publish_mjpg(this->imgs_[i], (char*)drawn_image_topic.str().c_str());   		
    	}
	}
}

void Classifier::carcmd_visualization(april_tags_gd_class_array_t* gd_class_array){

    std::pair<float, float> twist;
    std::pair<float, float> vel;
    for (int i = 0; i < this->batch_size_; i++){
        if (this->imgs_[i].empty()) break;
        twist = this->tf_probs2twist(gd_class_array->gd_array[i]); 
        vel = this->tf_probs2vel(gd_class_array->gd_array[i]); 
        std:: cout << " vel_left = " << vel.first << ", vel_right = " << vel.second << std::endl;
        std:: cout << " v = " << twist.first << ", omega = " << twist.second << std::endl;
        this->draw_arrowimage(twist.first, twist.second, i,cv::Scalar(0, 0, 255));
    }
}

std::pair<float , float> Classifier::tf_probs2vel(april_tags_gd_class_t gd_class){

    double vel_left = 0;
    double vel_right = 0;
    double speed = 0;
    double l_prob = 0.001;
    double s_prob = 0.001;
    double r_prob = 0.001;
    double max_speed = 1;
    double min_speed = 0.1;

	for (int i =  0; i < this->output_number_; i++){
        if(gd_class.preds[i].type == labels_[1]){
            s_prob = gd_class.preds[i].prob;
        }
        if(gd_class.preds[i].type == labels_[2]){
            r_prob = gd_class.preds[i].prob;
        }
        if(gd_class.preds[i].type == labels_[0]){
            l_prob = gd_class.preds[i].prob;
        }
	}
    
    if(gd_class.preds[0].type == labels_[1]){
        speed = max_speed * (1 + s_prob) / 2;      
        vel_left = (1-(r_prob*2)) * speed;
        vel_right = (1-(l_prob*2)) * speed;
    }else{
        speed = (r_prob+l_prob) * 0.5 * max_speed;
        vel_left = l_prob * speed * (max_speed-min_speed) + min_speed;
        vel_right = r_prob * speed * (max_speed-min_speed) + min_speed;
    }
     std::cout << "s_prob : " << s_prob << " r_prob : " << r_prob << " l_prob :" << l_prob << std::endl;
    std::cout << "speed : " << speed << " vel_left : " << vel_left << " vel_right :" << vel_right << std::endl;
    return std::make_pair(vel_left, vel_right);
	
}

std::pair<float , float> Classifier::tf_probs2twist(april_tags_gd_class_t gd_class){
	float v, omega;
	v = 0;
	omega = 0;
	float w_v[3] = {0,1,0};
	float w_omega[3] = {-1,0,1};
	for (int i =  0; i < this->output_number_; i++){
		for(int j = 0; j < labels_.size(); j++){
			if(gd_class.preds[i].type == labels_[j]){
				omega = omega - w_omega[j] *this->tf_probs2omega(gd_class.preds[i].prob);
				v = v + w_v[j] * gd_class.preds[i].prob;
			}
		}
	}
    return std::make_pair(v,omega);
}

float Classifier::tf_probs2omega(float prob){
	float o;
	if(prob <= 0.2)
        o  = prob /1.2;
    else if (prob >= 0.8)
        o = 1.8 + prob/(3-1.8);
    else
        o = 1.2 + prob/(1.8-1.2);
    return o ;
}

void Classifier::draw_arrowimage(float tmp_v,float tmp_omega,int j,cv::Scalar color){
	//std::stringstream drawn_image_topic;
    //drawn_image_topic << "image_with_motion_arrow";
 
	float angle,length;
    int start_x,start_y,end_x,end_y;
    float omega_max = 2.65;
    //cv::cvtColor(img,img_arrow,CV_BGR2RGB);  
    start_x = this->imgs_[j].size().width/2;
    start_y = this->imgs_[j].size().height;

    length = (tmp_v + 0.5) * 150;
    angle = tmp_omega / omega_max * 90;
    angle = angle / 180 * 3.1415;
    if (angle > 0)
        end_x = start_x - length * sin(abs(angle)) ;
    else
        end_x = start_x + length * sin(abs(angle)) ;
    end_y = start_y - length * cos(abs(angle)) ;
    end_x = int(end_x);
    end_y = int(end_y);
    int lineType = 8;
    int thickness = 10;
    cv::Point start(start_x,start_y);
    cv::Point end(end_x,end_y);
    cv::line(this->imgs_[j],start,end, color, thickness, lineType);
    //cv::cvtColor(img_arrow,img_arrow,CV_RGB2BGR);

    //std:: cout << " start = " << start_x << ", " << start_y << std::endl;
	//std:: cout << " end = " << end_x << ", " << end_y << std::endl;
    //cv_bridge_lcm_->publish_mjpg(this->imgs_[j], (char*)drawn_image_topic.str().c_str());


}


void Classifier::on_libbot_aux(const lcm_recv_buf_t* rbuf,
		const char* channel,
		const bot_core_image_t* msg,
		void* user_data) {
	(static_cast<Classifier *>(user_data))->on_libbot(msg);
}

void read_frames(string image_folder,
		vector<string> &frames_to_process, vector<string> &frames_names){

	boost::filesystem::path image_path(image_folder);
	boost::filesystem::recursive_directory_iterator end_it;


	//load files
	for (boost::filesystem::recursive_directory_iterator it(image_path); it != end_it; ++it) {

		//////////////////////////////////////////////////////////
		// read image
		if ((it->path().extension().string() == ".jpg"
				|| it->path().extension().string() == ".png"
						|| it->path().extension().string() == ".jpeg")
		){

			frames_to_process.push_back(it->path().string());
			frames_names.push_back(it->path().stem().string());
		}
	}
	// sort the image files
	sort(frames_to_process.begin(), frames_to_process.end());

}

void setup_signal_handlers(void (*handler)(int))
{
	struct sigaction new_action, old_action;
	memset(&new_action, 0, sizeof(new_action));
	new_action.sa_handler = handler;
	sigemptyset(&new_action.sa_mask);

	// Set termination handlers and preserve ignore flag.
	sigaction(SIGINT, NULL, &old_action);
	if (old_action.sa_handler != SIG_IGN)
		sigaction(SIGINT, &new_action, NULL);
	sigaction(SIGHUP, NULL, &old_action);
	if (old_action.sa_handler != SIG_IGN)
		sigaction(SIGHUP, &new_action, NULL);
	sigaction(SIGTERM, NULL, &old_action);
	if (old_action.sa_handler != SIG_IGN)
		sigaction(SIGTERM, &new_action, NULL);
}

int main(int argc, char** argv) {
	/*if (argc < 6) {
		std::cerr << "Usage: " << argv[0]
		    << " deploy.prototxt network.caffemodel"
		    << " mean.binaryproto labels.txt img.jpg"  << std::endl
		    << "Example 1 (file): " << std::endl
		    << argv[0]
		    << " models/bvlc_reference_caffenet/deploy.prototxt models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg"
		    << std::endl
			<< "Example 2 (folder): " << std::endl
			<< argv[0]
			<< " models/bvlc_reference_caffenet/deploy.prototxt models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt folder examples/images/"
			<< std::endl
			<< "Example 3 (lcm): " << std::endl
			<< argv[0]
			<< " models/bvlc_reference_caffenet/deploy.prototxt models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt lcm IMAGE_PICAMERA_wama"
			<< std::endl;
		return 1;
	}*/

	::google::InitGoogleLogging(argv[0]);

	string model_file;
	string trained_file;
	string mean_file;
	string label_file;
	string gpu_cpu;
 	string config_file_path;
 	string mode;
 	string image_channel;

 	double pred_threshold;
	double diff_threshold;
 	int batch_size;
 	int output_number;
 	int motion_visual;

	if(std::strcmp(argv[1], "configure") == 0){
		config_file_path = argv[2];
        std::cout << config_file_path << std::endl;
    	if (!boost::filesystem::exists(config_file_path)){
       		std::cout << "Can't find " << config_file_path<< std::endl;
    	}else{
			std::cout << "read: " << config_file_path << std::endl;

        	std::ifstream inputFile(config_file_path.c_str());

        	string line;
        	while (getline(inputFile, line)){
        		std::vector<std::string> x ;
        		boost::split(x, line, boost::is_any_of(":"));
				if(std::strcmp(x[0].c_str(), "model") == 0){
					model_file  = x[1];
				}else if(std::strcmp(x[0].c_str(), "trained_file") == 0){
					trained_file = x[1];
				}else if(std::strcmp(x[0].c_str(), "mean_file") == 0){
					mean_file = x[1];
				}else if(std::strcmp(x[0].c_str(), "label_file") == 0){
					label_file = x[1];
				}else if(std::strcmp(x[0].c_str(), "gpu_cpu") == 0){
					gpu_cpu = x[1];
				}else if(std::strcmp(x[0].c_str(), "output_number") == 0){
					output_number = atoi(x[1].c_str());
				}else if(std::strcmp(x[0].c_str(), "pred_threshold") == 0){
					pred_threshold = atof(x[1].c_str());
				}else if(std::strcmp(x[0].c_str(), "diff_threshold") == 0){
					diff_threshold = atof(x[1].c_str());
				}else if(std::strcmp(x[0].c_str(), "batch_size") == 0){
					batch_size  = atoi(x[1].c_str());
				}else if(std::strcmp(x[0].c_str(), "mode") == 0){
					mode  = x[1];
				}else if(std::strcmp(x[0].c_str(), "image_channel") == 0){
					image_channel  = x[1];	
				}else if(std::strcmp(x[0].c_str(), "motion_visual") == 0){
					motion_visual  = atoi(x[1].c_str());
				}
        		std::cout << x[0] << std::endl;
        		std::cout << x[1] << std::endl;
        	}
        }
    }

	Classifier classifier(model_file, trained_file, mean_file, label_file, gpu_cpu);
	classifier.set_output_number(output_number);
	classifier.set_threshold(pred_threshold, diff_threshold);
	classifier.set_batch_size(batch_size);
	classifier.set_motion_visual(motion_visual);
	if(std::strcmp(gpu_cpu.c_str(), "CPU") == 0){
		classifier.set_cpu(1);
	}
	/* set threshold */

    if(std::strcmp(mode.c_str(), "lcm") == 0){


		lcm_t* lcm = lcm_create(NULL);
		classifier.set_lcm(lcm, image_channel);
		
		std::cout << "---------- Prediction for "
				<< "lcm input channel " << image_channel << " ----------" << std::endl;
		
		classifier.run();
		
	}


}

