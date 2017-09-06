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
	void set_mean_setting(std::string mean_setting);
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
	void draw_prob_bar(april_tags_gd_class_array_t* gd_class_array);
	int get_predition_output_index(string output);

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
	int64_t end_time;
	bot_core_image_t_subscription_t* sub;
	// motion visualization
	bool motion_visual_;
    // repair bar label
	std::vector<std::string> bar_label_;
	std::vector<std::string> bar_label6_;
	// mean setting
	std::string mean_setting_;
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

    /* set bar label*/
	std::string barlabel;
	barlabel = "TL";
	this->bar_label_.push_back(barlabel);
	barlabel = "GS";
	this->bar_label_.push_back(barlabel);
	barlabel = "TR";
	this->bar_label_.push_back(barlabel);
	barlabel = "YBTL";
	this->bar_label6_.push_back(barlabel);
	barlabel = "YBGS";
	this->bar_label6_.push_back(barlabel);
	barlabel = "YBTR";
	this->bar_label6_.push_back(barlabel);
	barlabel = "FTTL";
	this->bar_label6_.push_back(barlabel);
	barlabel = "FTGS";
	this->bar_label6_.push_back(barlabel);
	barlabel = "FTTR";
	this->bar_label6_.push_back(barlabel);

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
	if(boost::filesystem::exists(mean_file)){
		//std::cout << " get mean file" << std::endl;
		//SetMean(mean_file);
	}

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
void Classifier::set_mean_setting(std::string mean_setting){
	this->mean_setting_ = mean_setting;
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
	//channel_mean = cv::Scalar(136, 145, 154); // added by brian
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
        //std::cout << "original image mean: " << cv::mean(sample) << std::endl;

        /* resize smaple image to fit input channel*/
        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
          cv::resize(sample, sample_resized, input_geometry_);
        else
          sample_resized = sample;
        cv::Mat sample_float;
        if (num_channels_ == 3)
          sample_resized.convertTo(sample_float, CV_32FC3);
        else
          sample_resized.convertTo(sample_float, CV_32FC1);

        //std::cout << "resized image mean: " << cv::mean(sample_float) << std::endl;

        /* mean substraction for guidedog */
        if(this->mean_setting_.compare("BVLC") == 0){
        	//std::cout << "use mean setting: " << this->mean_setting_ << std::endl;
            cv::Mat m = cv::Mat(input_geometry_.height, input_geometry_.width, CV_32FC3, cv::Scalar(136, 145, 154));
            cv::subtract(sample_float, m, sample_float);
        }else if(this->mean_setting_.compare("PDNN2_COLOR") == 0){
        	//std::cout << "use mean setting: " << this->mean_setting_ << std::endl;
         	cv::Mat m = cv::Mat(input_geometry_.height, input_geometry_.width, CV_32FC3, cv::Scalar(128, 128, 128));
         	cv::subtract(sample_float, m, sample_float);
         	sample_float = sample_float * 0.0078125;
		}else if(this->mean_setting_.compare("PDNN2") == 0){
        	//std::cout << "use mean setting: " << this->mean_setting_ << std::endl;
         	cv::Mat m = cv::Mat(input_geometry_.height, input_geometry_.width, CV_32FC1, cv::Scalar(128));
         	cv::subtract(sample_float, m, sample_float);
         	sample_float = sample_float * 0.0078125;}

        //std::cout << "mean substraction mean: " << cv::mean(sample_float) << std::endl;

        /* old normalize */
        //cv::Mat sample_normalized;
        //cv::subtract(sample_float, mean_, sample_normalized);
      	//cv::normalize(sample_float, sample_float, 0, 255, NORM_MINMAX, CV_32FC1);

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
   
	end_time = bot_timestamp_now();
    //std::cout << "process time: " << end_time - start_time << std::endl;

	for (int j=0; j < this->batch_size_; j++)
	{
		for (int k = 0; k < this->output_number_; k++)
		{
			Prediction p = predictions[j][k];
			//std::cout << p.second << " - " << p.first << std::endl;

            types[j][k].type = (char*)p.first.c_str();
            types[j][k].prob = predictions[j][k].second;                 
		}	
		//std::cout << "--------------------------" << std::endl;
		gd_array[j].output_number = this->output_number_;
		gd_array[j].preds = types[j];
	}
    gd_class_array.gd_array = gd_array;
	gd_class_array.batch_size = this->batch_size_;
	april_tags_gd_class_array_t_publish(lcm_, "gd_class_array", &gd_class_array);
	if(this->motion_visual_){
		/* draw motion arrow */
//    	this->carcmd_visualization(&gd_class_array);
    	/* draw probability bar */
        this->draw_prob_bar(&gd_class_array);
    	std::stringstream drawn_image_topic;
    	/* publish drawn iamges */
    	drawn_image_topic << "image_with_motion_arrow_and_probability_bar";
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
        /* calculate v and omega from predictions*/
        twist = this->tf_probs2twist(gd_class_array->gd_array[i]); 
        //std:: cout << " v = " << twist.first << ", omega = " << twist.second << std::endl;
        /* calculate velocity_left and velocity_right from predictions*/       
        //vel = this->tf_probs2vel(gd_class_array->gd_array[i]); 
        //std:: cout << " vel_left = " << vel.first << ", vel_right = " << vel.second << std::endl;
        /* draw motion arrow on sample images */
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
     //std::cout << "s_prob : " << s_prob << " r_prob : " << r_prob << " l_prob :" << l_prob << std::endl;
    //std::cout << "speed : " << speed << " vel_left : " << vel_left << " vel_right :" << vel_right << std::endl;
    return std::make_pair(vel_left, vel_right);
	
}

std::pair<float , float> Classifier::tf_probs2twist(april_tags_gd_class_t gd_class){
	float v, omega;
	v = 0;
	omega = 0;
	std::vector<float> w_omega;
	if(this->output_number_ == 3){
		w_omega.push_back(-1);
		w_omega.push_back(0);
		w_omega.push_back(1);
	}else if(this->output_number_ == 5){
		w_omega.push_back(-1);
		w_omega.push_back(0);
		w_omega.push_back(0);
		w_omega.push_back(0);
		w_omega.push_back(1);
	}
	for (int i =  0; i < this->output_number_; i++){
		for(int j = 0; j < labels_.size(); j++){
			if(gd_class.preds[i].type == labels_[j]){
				omega = omega + w_omega[j] * this->tf_probs2omega(gd_class.preds[i].prob);
			}
			v = 0.38;
		}
	}
    return std::make_pair(v,omega);
}

float Classifier::tf_probs2omega(float prob){
	float o, omega_scalar, sig_interval, alpha;
	omega_scalar = 9;
	sig_interval = 10;
	alpha = 0.4;
	/* sigmoid transfer function */
	prob = -sig_interval + prob * 2 * sig_interval;
	o = 1 / (1 + std::exp(-prob * alpha)); 
    o = o * omega_scalar;
	/*if(prob <= 0.2)
        o  = prob /1.2;
    else if (prob >= 0.8)
        o = 1.8 + prob/(3-1.8);
    else
        o = 1.2 + prob/(1.8-1.2);*/
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

int Classifier::get_predition_output_index(std::string output){
	int index = 0;

    //output.erase(output.length()-1);
	for(int i = 0; i <  this->output_number_; i++){
		//std::cout << this->labels_[i] <<std::endl;
		//std::cout << output <<std::endl;
		
		if(this -> labels_[i].compare(output) == 0){
			//std::cout << this->labels_[i] <<std::endl;
			index = i;
		}
	}
	return index;
}

void Classifier::draw_prob_bar(april_tags_gd_class_array_t* gd_class_array){

    std::pair<float, float> twist;
    std::pair<float, float> vel;
    std::string label;
    cv::Point bar_start;  
    cv::Point bar_end;
    cv::Point bar_bg_end;
    cv::Point label_point;
    int shift;
    int bar_width = 50;
    int bar_left_bound = int ( (this->imgs_[0].size().width) / 2 - (bar_width * this->output_number_ / 2));
    int bar_height_bound = int (this->imgs_[0].size().height);

    for (int i = 0; i < this->batch_size_; i++){
    	/* check images exit */
        if (this->imgs_[i].empty()) break;  
        for(int j = 0; j < this->output_number_; j++){
        	/* get bar left buttom and right top points */
            shift = this->get_predition_output_index(gd_class_array->gd_array[i].preds[j].type);
			/* fix shift for TL GS TR*/
			if(this->labels_.size() == 3){
				shift = 2 - shift;
			}else if(this->labels_.size() == 6){
			    if(shift <= 2){
				    shift = 2 - shift;
				}else{
				    shift = 8 - shift;
				}
			}
            bar_start = cv::Point(bar_left_bound + shift * bar_width, bar_height_bound);
            bar_end = cv::Point(bar_left_bound + (shift + 1) * bar_width, bar_height_bound - int(gd_class_array->gd_array[i].preds[j].prob * 100));
            bar_bg_end = cv::Point(bar_left_bound + (shift + 1) * bar_width, bar_height_bound -100);
            /* draw probability bar */
            bar_start = cv::Point(bar_left_bound + shift * bar_width, bar_height_bound);
            cv::rectangle(this->imgs_[i], bar_start, bar_bg_end, cv::Scalar(255, 255, 255), -1, 8, 0);
            cv::rectangle(this->imgs_[i], bar_start, bar_end, cv::Scalar(0, 0, 255), -1, 8, 0);
            /* write classes labels */
            //label = this->labels_[shift];
            //label.erase(label.length()-1);
            label = this->bar_label6_[shift];
			label_point = cv::Point(bar_left_bound + shift * bar_width + 10, bar_height_bound - 10);
            cv::putText(this->imgs_[i],label,label_point,0,0.5,cv::Scalar(255, 170, 0),2);
        }
    }
    
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

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0]  << std::endl
		    << "Example 1 (lcm): " << std::endl
		    << " ./build/lcm/cpp_classification/lcm_classification_batch_image_gulidedog.bin configure /home/robotvision/icra2018-guidedog/config/VB/model_config_PDNN2_YB5_C_S_color.txt"
		    << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);

	string model_file;
	string trained_file;
	string mean_file;
	string label_file;
	string gpu_cpu;
 	string config_file_path;
 	string mode;
 	string image_channel;
 	string image_folder;
 	string test_file;
 	string mean_setting;

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
				}else if(std::strcmp(x[0].c_str(), "image_folder") == 0){
					image_folder  = x[1];
				}else if(std::strcmp(x[0].c_str(), "test_file") == 0){
					test_file  = x[1];
				}else if(std::strcmp(x[0].c_str(), "mean_setting") == 0){
					mean_setting  = x[1];
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
	classifier.set_mean_setting(mean_setting);
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


    if(std::strcmp(mode.c_str(), "test_file") == 0){

        /* set up parameter */
		int image_folder_image_number = 0;
		vector<string> image_path;
		vector<int> gt;
		std::vector<cv::Mat> image_folder_imgs;
		std::vector< std::vector<Prediction> > predictions;
	    int testing_data_number;
        int hit[output_number];
        int non_hit[output_number]; 
        for (int i = 0; i < output_number; ++i){
            hit[i] = 0;
            non_hit[i] = 0;
        }             

        /* load images path and groundtruth */
		if (!boost::filesystem::exists(test_file)){
        	std::cout << "Can't find " << test_file << std::endl;
    	}else{
            std::cout << "read: " << test_file << std::endl;
            std::ifstream inputFile(test_file.c_str());
            string line;
            while (getline(inputFile, line))
            {
                std::vector<std::string> strs;
                boost::split(strs, line, boost::is_any_of(" "));
                if(strs.size() < 1){
                    break;
                }
                std::cout << strs[0] << strs[1] << std::endl;
                image_path.push_back(strs[0]);
                gt.push_back(atoi(strs[1].c_str()));;
		    }
		}
		testing_data_number = gt.size();
        /* load label list */
        std::vector<string> label_list;
	    std::ifstream labels_out(label_file.c_str());
	    CHECK(labels_out) << "Unable to open labels file " << label_file;
	    string line;
	    while (std::getline(labels_out, line))
		    label_list.push_back(string(line));
		for (int i = 0; i < 5; i++){
			std::cout << label_list[i] << std::endl;
		}         
		/* output result txt file */
		std::ofstream testing_reuslt;
		std::stringstream ss;
		ss << "testing_reuslt.txt";
		std::cout << ss.str().c_str() << " created"<< std::endl;
        testing_reuslt.open(ss.str().c_str());

        /* do prediction */
		for(int f = 0; f < image_path.size(); f++){
            image_folder_image_number ++;
			std::cout << "---------- Prediction for "
					<< image_path[f] << " ----------" << std::endl;
            cv::Mat img = cv::imread(image_path[f], -1);
            //cv::Mat img = cv::imread(image_path[f], 0);
            image_folder_imgs.push_back(img);
            if(image_folder_imgs.size() == batch_size || image_folder_image_number == testing_data_number){
                if(image_folder_image_number == testing_data_number){
                	int delta_image = batch_size - image_folder_imgs.size();
                    for(int j = 0; j < delta_image; j++){
                        image_folder_imgs.push_back(img);
                    }
                    f = f + delta_image;
                }

                unsigned long t_init = GetTickCount();
                predictions = classifier.ClassifyBatch(image_folder_imgs, output_number);
                unsigned long t_last = GetTickCount();
	            std::cout << "classifying time: " << t_last-t_init<< "ms" << std::endl;
	            for (int j=0; j < batch_size; j++){
	            	int index = f - batch_size + 1 + j;
	            	std::cout << "image number: " << index << std::endl;
	            	if( index < testing_data_number){
	            		string gt_label = label_list[gt[index]];
	            		testing_reuslt << index << "," << image_path[index] << "," << gt_label.erase(gt_label.length() - 1) << ',';
		            	for (int k = 0; k < output_number; k++){
			            	Prediction p = predictions[j][k];
			            	std::cout << p.second << " - " << p.first << std::endl; 
			            	testing_reuslt << p.second << "," << p.first.erase(p.first.length() - 1) << ',';   
		            	}	
		            	if(label_list[gt[index]].compare(predictions[j][0].first) == 0){
                        	hit[gt[index]] ++;
                        	testing_reuslt << "hit" << std::endl;
		            	}else{
                        	non_hit[gt[index]] ++;
                        	testing_reuslt << "non_hit" << std::endl;
		            	}
                    	for (int i = 0; i < output_number; ++i){
                        	std::cout << label_list[i] << " hit: " << hit[i] << " non_hit: " << non_hit[i] << std::endl;
                    	}  	 
                    }           
		        std::cout << "--------------------------" << std::endl;

              	}
              	image_folder_imgs.clear();
            }
		}
		testing_reuslt << "hit_number";
		for(int i = 0; i < output_number; i++){
			testing_reuslt << "," << hit[i];
		}
		testing_reuslt << std::endl;
		testing_reuslt << "non_hit_number";
		for(int i = 0; i < output_number; i++){
			testing_reuslt << "," << non_hit[i];
		}
		testing_reuslt << std::endl;
		testing_reuslt << "accuracy";
		for(int i = 0; i < output_number; i++){
			testing_reuslt << "," << (double) hit[i] / (hit[i] + non_hit[i]);
		}
		testing_reuslt << std::endl;
		testing_reuslt << "average accuracy";
		int image_sum = 0;
		int hit_sum = 0;
		for(int i = 0; i < output_number; i++){
			image_sum += hit[i];
			image_sum += non_hit[i];
			hit_sum += hit[i];
		}
		testing_reuslt << "," << (double) hit_sum  / image_sum;
		testing_reuslt << std::endl;		
		


	}


}

