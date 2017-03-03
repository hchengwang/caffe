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

#include <lcm/lcm.h>
#include <bot_core/bot_core.h>
#include <cv_bridge_lcm/rv-cv-bridge-lcm.h>
#include <lcmtypes/bot_core_image_t.h>
#include <bot_lcmgl_client/lcmgl.h>
#include <GL/gl.h>
#include <lcmtypes/april_tags_tag_text_detection_t.h>
#include <lcmtypes/april_tags_tag_text_detections_t.h>
#include <lcmtypes/april_tags_quad_proposal_t.h>
#include <lcmtypes/april_tags_quad_proposals_t.h>
#include "lcmtypes/april_tags_caffe_class_t.h"
#include "lcmtypes/april_tags_caffe_class_array_t.h"
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

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
    std::vector< vector<Prediction> > ClassifyBatch(const vector< cv::Mat > imgs, int N = 5);
	void set_lcm(lcm_t* lcm, std::string image_channel);
	void set_cpu(int is_cpu);
	void set_threshold(double pred_threshold, double diff_threshold);
	void set_batch_size(int batch_size);
	void set_output_number(int output_number);
	void draw_annotation(cv::Mat &im, cv::Rect bbox, string anno,cv::Scalar color, int shift_x);
	void run();
	void finish();

	// Message Handler
	void on_libbot(const bot_core_image_t *msg);
	void on_libbot_daniel(const bot_core_image_t *msg);
	static void on_libbot_aux(const lcm_recv_buf_t* rbuf,
			const char* channel,
			const bot_core_image_t* msg,
			void* user_data);

	void on_tag_text_detections(const april_tags_tag_text_detections_t* msg);
	void on_tag_text_detection(april_tags_tag_text_detection_t msg);
	static void on_tag_text_detections_aux(const lcm_recv_buf_t* rbuf,
			const char* channel,
			const april_tags_tag_text_detections_t* msg,
			void* user_data);

	void on_quad_proposals(const april_tags_quad_proposals_t* msg);
	static void on_quad_proposals_aux(const lcm_recv_buf_t* rbuf,
			const char* channel,
			const april_tags_quad_proposals_t* msg,
			void* user_data);

private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);
	std::vector<float> PredictBatch(const vector< cv::Mat > imgs);
	
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);
	
	void Preprocess(const cv::Mat& img,
			std::vector<cv::Mat>* input_channels);
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
	// caffe class;
	april_tags_caffe_class_t caffe_class_;
	// threshold 
	double pred_threshold_;
	double diff_threshold_;
	// batch size
	int batch_size_;
	// image;
	cv::Mat img_;
	cv::Mat img_rgb_;
	// get image;
	int image_number_;
	bool get_image_;
	// output channel;
	int output_number_;
	// LCM Message Channels
	std::string image_channel_;

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
	batch_size_ = 5;
	this->image_number_ = 0;
	this->get_image_ = false;
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
        
	bot_core_image_t_subscription_t* sub = bot_core_image_t_subscribe(
			lcm, image_channel.c_str(), Classifier::on_libbot_aux, this);

	const char* atags_channel = "TAG_TEXT_DETECTIONS";
	april_tags_tag_text_detections_t_subscribe(
			lcm, atags_channel, Classifier::on_tag_text_detections_aux, this);

	const char* quads_channel = "QUAD_PROPOSALS";
	april_tags_quad_proposals_t_subscribe(
			lcm, quads_channel, Classifier::on_quad_proposals_aux, this);

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
	batch_size_ = batch_size;
}
void Classifier::set_output_number(int output_number){
	this->output_number_ = output_number;
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
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

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

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;

	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);
	std::cout << "input size : " << input_channels.size() << std::endl;
	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

std::vector< float >  Classifier::PredictBatch(const vector< cv::Mat > imgs) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  input_layer->Reshape(batch_size_, num_channels_,
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
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

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


void Classifier::Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

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
	//**important modified for no mean file(comment the below two lines)
	//cv::Mat sample_normalized;
	//cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	//**important modified for no mean file(change sample_normalized to sample_float)	
	cv::split(sample_float, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
			== net_->input_blobs()[0]->cpu_data())
	<< "Input channels are not wrapping the input layer of the network.";
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

void Classifier::on_libbot_daniel(const bot_core_image_t* image_msg) {
	std::cout << "callback" << std::endl;
	start_time = bot_timestamp_now();
//Modified for text detection by Daniel
	
	cv::Mat im_rgb = cv::Mat::zeros(image_msg->height, image_msg->width, CV_8UC3);
	if(image_msg->pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG){
	//jpegijg_decompress_8u_rgb (image_msg->data, image_msg->size,im_rgb.data, image_msg->width, image_msg->height, image_msg->width * 3);
	}
	cv::Mat im_rgb_flipx, im_rgb_flipy, im_thres;
	//cv::flip(im_rgb, im_rgb_flipx, 0);
	//cv::flip(im_rgb_flipx, im_rgb_flipy, 1);
	cv::inRange(im_rgb, cv::Scalar(103, 142, 45), cv::Scalar(123, 179, 119), im_thres);
	cv::resize(im_thres, im_thres, cv::Size(160,120));
	Mat element = getStructuringElement(MORPH_RECT,Size(2,2));
	cv::dilate(im_thres, im_thres, element);
	//cv_bridge_lcm_->publish_gray(im_thres, (char*)"IMAGE_THRES");
	vector<vector<Point> > contours;
  	vector<Vec4i> hierarchy;
	findContours( im_thres, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) ); 
	vector<RotatedRect> minRect( contours.size() );
	int rect_num=0;
	//std::cout << "contour size" << contours.size() << std::endl;
	for( int i = 0; i < contours.size(); i++ ){
	//std::cout << contourArea(contours[i]) << std::endl;
	if(contourArea(contours[i])>200){
	drawContours( im_thres, contours, i, 100, 1, 8, vector<Vec4i>(), 0, Point() );	
	minRect[rect_num] = minAreaRect( Mat(contours[i]) );
	
	rect_num++;
	}
	}
	Point2f vertices[4];
	
	for (int index = 0; index < rect_num; index++){
	minRect[index].points(vertices);
	std::cout << "////////" << std::endl << vertices[0].x << " " << vertices[1].x << " " << vertices[2].x << " " << vertices[3].x << std::endl;
	std::cout << vertices[0].y << " " << vertices[1].y << " " << vertices[2].y << " " << vertices[3].y  << std::endl;	
	for (int i = 0; i < 4; i++)
	line(im_thres, vertices[i], vertices[(i+1)%4], 100);
	}
	//cv_bridge_lcm_->publish_gray(im_thres, (char*)"IMAGE_BOX");

	vector<Point2f> inputpts, outputpts;
	float centerx=(vertices[0].x+vertices[1].x+vertices[2].x+vertices[3].x)/4, centery=(vertices[0].y+vertices[1].y+vertices[2].y+vertices[3].y)/4;
	for(int i=0; i<4; i++){
	if((vertices[i].x<=centerx)&&(vertices[i].y<=centery))inputpts.push_back(Point2f(vertices[i].x*4,vertices[i].y*4));
	}	
	for(int i=0; i<4; i++){
	if((vertices[i].x<=centerx)&&(vertices[i].y>centery))inputpts.push_back(Point2f(vertices[i].x*4,vertices[i].y*4));
	}
for(int i=0; i<4; i++){
	if((vertices[i].x>centerx)&&(vertices[i].y>centery))inputpts.push_back(Point2f(vertices[i].x*4,vertices[i].y*4));
	}
for(int i=0; i<4; i++){
	if((vertices[i].x>centerx)&&(vertices[i].y<=centery))inputpts.push_back(Point2f(vertices[i].x*4,vertices[i].y*4));
	}

	outputpts.push_back(Point2f(float(0), float(0)));
    	outputpts.push_back(Point2f(float(0), float(31)));
    	outputpts.push_back(Point2f(float(99), float(31)));
    	outputpts.push_back(Point2f(float(99), float(0)));
	Mat transMat = findHomography( inputpts, outputpts, 0 );
   	Mat outputgray = Mat::zeros(32, 100, CV_8UC1);
    	Mat output;
	
    	warpPerspective(im_rgb, output, transMat, Size(100, 32));
	cvtColor(output, outputgray, CV_BGR2GRAY);
	for(int i=0; i<4; i++)	
	circle(im_rgb, Point2f(vertices[i].x*4,vertices[i].y*4), 2, Scalar(0, 0, 200), -1);
	//cv_bridge_lcm_->publish_mjpg(im_rgb_flipy, (char*)"IMAGE_flip");	
	//cv_bridge_lcm_->publish_mjpg(output, (char*)"IMAGE_cropped");	
	std::cout<<"region_prop time"<<bot_timestamp_now()-start_time<<"us"<<std::endl;
	
	unsigned long t_init = GetTickCount();
	std::vector<Prediction> predictions = this->Classify(output);
	
	for (size_t i = 0; i < predictions.size(); ++i) {
			Prediction p = predictions[i];
			std::cout << std::fixed << i << " " <<std::setprecision(4) << p.second << " - \""<< p.first << "\"" << std::endl;
		}
	unsigned long t_last = GetTickCount();
	std::cout << "classifying time: " << t_last-t_init<< "ms" << std::endl;

	
	char temp0[50], temp1[50], temp2[50], temp3[50], temp4[50];
	strcpy(temp0, predictions[0].first.c_str());
	strcpy(temp1, predictions[1].first.c_str());
	strcpy(temp2, predictions[2].first.c_str());
	strcpy(temp3, predictions[3].first.c_str());
	strcpy(temp4, predictions[4].first.c_str());
	
	
	caffe_class_.class0=temp0;
	caffe_class_.class1=temp1;
	caffe_class_.class2=temp2;
	caffe_class_.class3=temp3;
	caffe_class_.class4=temp4;
	


	april_tags_caffe_class_t_publish(lcm_, "caffe_class", &caffe_class_);
//	cv::Mat im_rgb = cv::Mat::zeros(image_msg->height, image_msg->width, CV_8UC3);
//	cv::Mat im_vis = cv::Mat::zeros(image_msg->height, image_msg->width, CV_8UC3);
//
//	if(image_msg->pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG){
//		// jpeg decompress
//	    jpegijg_decompress_8u_rgb (image_msg->data, image_msg->size,
//	    		im_rgb.data, image_msg->width, image_msg->height, image_msg->width * 3);
//	}else {
//		im_rgb.data = image_msg->data;
//	}
//        //brian
//        this->image = im_rgb;
//        //
//	cv::Mat im_rect = cv::Mat::zeros(im_rgb.rows, im_rgb.cols, CV_8UC1);
//	cv::cvtColor(im_rgb, im_rect, CV_RGB2GRAY);
	// TODO: rectify image
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
	this->get_image_ = true;
}


void Classifier::on_tag_text_detection (
		const april_tags_tag_text_detection_t detection) {
}

void Classifier::on_tag_text_detections (
		const april_tags_tag_text_detections_t* detections) {
	std::cout << "bbox" << std::endl;
}

void Classifier::on_quad_proposals(const april_tags_quad_proposals_t* proposals_msg) {
	 // measure process time
	int64_t start_time;
	start_time = bot_timestamp_now();
    // return if image not received yet
	std::cout << "get quad: " << proposals_msg->n << std::endl;
	cv::Mat im_rgb_to_pub; 
	im_rgb_to_pub = this->img_rgb_;
	if(!this->get_image_){
		std::cout << "non get image" << std::endl;
		return;
	}
	if(proposals_msg->predicted == 0){
    	april_tags_quad_proposals_t_publish(lcm_, "QUAD_PROPOSALS_RECT", proposals_msg);

   		stringstream ss_out;
    	ss_out << "IMAGE_PICAMERA_caffe";
    	cv_bridge_lcm_->publish_mjpg(im_rgb_to_pub, (char*)ss_out.str().c_str());
		std::cout << "process image time: " << (bot_timestamp_now() - start_time) << std::endl;
		return;	
	}

    cv::Rect crop_rect;
    std::vector<cv::Mat> imgs;
    std::vector< std::vector<Prediction> > predictions;

    int hit[proposals_msg->n];
    april_tags_caffe_class_t caffe_array[proposals_msg->n];
    april_tags_caffe_class_array_t caffe_class_array;

    int loop = 0;

    for(int i = 0; i < proposals_msg->n; i++){
		crop_rect.x = proposals_msg->proposals[i].x;
		crop_rect.y = proposals_msg->proposals[i].y;
    	crop_rect.width = proposals_msg->proposals[i].width;
       	crop_rect.height = proposals_msg->proposals[i].height;
       	std::cout << crop_rect.x << "," << crop_rect.y << "," << crop_rect.width << "," << crop_rect.height << std::endl;
       	std::stringstream image_name;

       	// write image
       	//image_name << "crop_image/" << this->image_number_ << ".jpg";
       	//cv::imwrite(image_name.str(), this->img_(crop_rect));
       	//this->image_number_ = this->image_number_ + 1;

	 	if(imgs.size() < batch_size_){
      		imgs.push_back(this->img_(crop_rect));	
    	}
    	if( i == (proposals_msg->n - 1)){
    		std::cout << "rest images" << std::endl;
    		predictions = this->ClassifyBatch(imgs, this->output_number_);

			for (int j = 0; j < predictions.size(); j++) {
				std::cout << "logfiff : " << std::log10(predictions[j][0].second/predictions[j][1].second) << std::endl;
				if(predictions[j][0].second <= this->pred_threshold_){
					std::cout << "under prediction threshold" << std::endl;
					hit[j+loop*10] = 0;				
					caffe_array[j+loop*10].class0=(char*)predictions[j][0].first.c_str();
					caffe_array[j+loop*10].class1=(char*)predictions[j][1].first.c_str();
					caffe_array[j+loop*10].class2=(char*)predictions[j][2].first.c_str();
					caffe_array[j+loop*10].class3=(char*)predictions[j][3].first.c_str();
					caffe_array[j+loop*10].class4=(char*)predictions[j][4].first.c_str();
									for (int k = 0; k < 5; k++) {
					//Prediction p = predictions[j][k];
					//std::cout << p.second << " - " << p.first << std::endl;
				}
					continue;
				}
				if(std::log10(predictions[j][0].second/predictions[j][1].second) <= this->diff_threshold_){
					std::cout << "non enough difference" << std::endl;
					hit[j+loop*10] = 0;				
					caffe_array[j+loop*10].class0=(char*)predictions[j][0].first.c_str();
					caffe_array[j+loop*10].class1=(char*)predictions[j][1].first.c_str();
					caffe_array[j+loop*10].class2=(char*)predictions[j][2].first.c_str();
					caffe_array[j+loop*10].class3=(char*)predictions[j][3].first.c_str();
					caffe_array[j+loop*10].class4=(char*)predictions[j][4].first.c_str();
									for (int k = 0; k < 5; k++) {
					//Prediction p = predictions[j][k];
					//std::cout << p.second << " - " << p.first << std::endl;
				}
					continue;
				}
				for (int k = 0; k < 5; k++) {
					Prediction p = predictions[j][k];
					std::cout << p.second << " - " << p.first << std::endl;
				}
					hit[j+loop*10] = 1;				
					caffe_array[j+loop*10].class0=(char*)predictions[j][0].first.c_str();
					caffe_array[j+loop*10].class1=(char*)predictions[j][1].first.c_str();
					caffe_array[j+loop*10].class2=(char*)predictions[j][2].first.c_str();
					caffe_array[j+loop*10].class3=(char*)predictions[j][3].first.c_str();
					caffe_array[j+loop*10].class4=(char*)predictions[j][4].first.c_str();
				
			}
			imgs.clear();
    	}

    	if(imgs.size() == batch_size_){
    		std::cout << "up to batch size images" << std::endl;
    		predictions = this->ClassifyBatch(imgs, this->output_number_);

			for (int j = 0; j < predictions.size(); j++) {
				std::cout << "logfiff : " << std::log10(predictions[j][0].second/predictions[j][1].second) << std::endl;
				if(predictions[j][0].second <= this->pred_threshold_){
					std::cout << "under prediction threshold" << std::endl;
					hit[j+loop*10] = 0;				
					caffe_array[j+loop*10].class0=(char*)predictions[j][0].first.c_str();
					caffe_array[j+loop*10].class1=(char*)predictions[j][1].first.c_str();
					caffe_array[j+loop*10].class2=(char*)predictions[j][2].first.c_str();
					caffe_array[j+loop*10].class3=(char*)predictions[j][3].first.c_str();
					caffe_array[j+loop*10].class4=(char*)predictions[j][4].first.c_str();
					continue;
				}
				if(std::log10(predictions[j][0].second/predictions[j][1].second) <= this->diff_threshold_){
					std::cout << "non enough difference" << std::endl;
					hit[j+loop*10] = 0;				
					caffe_array[j+loop*10].class0=(char*)predictions[j][0].first.c_str();
					caffe_array[j+loop*10].class1=(char*)predictions[j][1].first.c_str();
					caffe_array[j+loop*10].class2=(char*)predictions[j][2].first.c_str();
					caffe_array[j+loop*10].class3=(char*)predictions[j][3].first.c_str();
					caffe_array[j+loop*10].class4=(char*)predictions[j][4].first.c_str();
					continue;
				}

				for (int k = 0; k < 5; k++) {
					Prediction p = predictions[j][k];
					std::cout << p.second << " - " << p.first << std::endl;
				}
				hit[j+loop*10] = 1;
				caffe_array[j+loop*10].class0=(char*)predictions[j][0].first.c_str();
				caffe_array[j+loop*10].class1=(char*)predictions[j][1].first.c_str();
				caffe_array[j+loop*10].class2=(char*)predictions[j][2].first.c_str();
				caffe_array[j+loop*10].class3=(char*)predictions[j][3].first.c_str();
				caffe_array[j+loop*10].class4=(char*)predictions[j][4].first.c_str();
			}
			imgs.clear();
			loop = loop + 1;
    	}

    }

    caffe_class_array.n = proposals_msg->n;
	caffe_class_array.hit = hit;
	caffe_class_array.caffe_array = caffe_array;
	for(int i = 0; i < caffe_class_array.n; i++){
		std::cout << caffe_class_array.caffe_array[i].class0 << std::endl;
	}
    april_tags_quad_proposals_t_publish(lcm_, "QUAD_PROPOSALS_RECT", proposals_msg);
    april_tags_caffe_class_array_t_publish(lcm_, "caffe_class_array", &caffe_class_array);

    cv::Rect draw_rect;
    for(int i = 0; i < proposals_msg->n; i++){
		draw_rect.x = proposals_msg->proposals[i].x;
		draw_rect.y = proposals_msg->proposals[i].y;
    	draw_rect.width = proposals_msg->proposals[i].width;
       	draw_rect.height = proposals_msg->proposals[i].height;
       	if(caffe_class_array.hit[i] == 0){
    		this->draw_annotation(im_rgb_to_pub, draw_rect, "", cv::Scalar(0, 0, 255), 0);
    	}
    }
    for(int i = 0; i < proposals_msg->n; i++){
		draw_rect.x = proposals_msg->proposals[i].x;
		draw_rect.y = proposals_msg->proposals[i].y;
    	draw_rect.width = proposals_msg->proposals[i].width;
       	draw_rect.height = proposals_msg->proposals[i].height;
       	if(caffe_class_array.hit[i] == 1){
       		stringstream caffe_turn;
			// assign to string and remove \n
			std::string text_prediction_modified;
			text_prediction_modified = caffe_class_array.caffe_array[i].class0;
			text_prediction_modified.erase(text_prediction_modified.length()-1);
			/*
			if(std::strcmp(text_prediction_modified.c_str(), "LEONARD") == 0){
				caffe_turn << text_prediction_modified << ": turn right";
			}
			if(std::strcmp(text_prediction_modified.c_str(), "RUS") == 0){
				caffe_turn << text_prediction_modified << ": turn left";
			}
			if(std::strcmp(text_prediction_modified.c_str(), "IAGNEMA") == 0){
				caffe_turn << text_prediction_modified << ": turn right";
			}
			if(std::strcmp(text_prediction_modified.c_str(), "HOW") == 0){
				caffe_turn << text_prediction_modified << ": turn left";
			}
    		this->draw_annotation(im_rgb_to_pub, draw_rect, caffe_class_array.caffe_array[i].class0, cv::Scalar(0, 255, 0), 0);
    		*/
    		std::cout << "get hit" << std::endl;
    		this->draw_annotation(im_rgb_to_pub, draw_rect, text_prediction_modified, cv::Scalar(0, 255, 0), 0);
    	}
    }

    stringstream ss_out;
    ss_out << "IMAGE_PICAMERA_caffe";
    cv_bridge_lcm_->publish_mjpg(im_rgb_to_pub, (char*)ss_out.str().c_str());

	std::cout << "process image time: " << (bot_timestamp_now() - start_time) << std::endl;

}

void Classifier::draw_annotation(cv::Mat &im, cv::Rect bbox, string anno,
        cv::Scalar color, int shift_x){

    rectangle( im, bbox.tl(), bbox.br(), color, 2, 8, 0 );

    // put text
    string text = anno;
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;

    int baseline=0;
    cv::Size textSize = cv::getTextSize(text, fontFace,
            fontScale, thickness, &baseline);
    baseline += thickness;

    // center the text
    cv::Point textOrg(bbox.x + shift_x, bbox.y - 8);

    // draw the box
    cv::rectangle(im, textOrg + cv::Point(0, baseline),
            textOrg + cv::Point(textSize.width, -textSize.height),
            color, CV_FILLED);

    // then put the text itself
    cv::putText(im, text, textOrg, fontFace, fontScale,
            cv::Scalar::all(0), thickness, 8);
}




void Classifier::on_libbot_aux(const lcm_recv_buf_t* rbuf,
		const char* channel,
		const bot_core_image_t* msg,
		void* user_data) {
	(static_cast<Classifier *>(user_data))->on_libbot(msg);
}

void Classifier::on_tag_text_detections_aux( const lcm_recv_buf_t* rbuf,
		const char* channel,
		const april_tags_tag_text_detections_t* detections,
		void* user_data) {
	(static_cast<Classifier *>(user_data))->on_tag_text_detections(detections);
}

void Classifier::on_quad_proposals_aux( const lcm_recv_buf_t* rbuf,
		const char* channel,
		const april_tags_quad_proposals_t* proposals,
		void* user_data) {
	(static_cast<Classifier *>(user_data))->on_quad_proposals(proposals);
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
	if (argc < 6) {
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
	}

	::google::InitGoogleLogging(argv[0]);

	string model_file   = argv[1];
	string trained_file = argv[2];
	string mean_file    = argv[3];
	string label_file   = argv[4];
	string gpu_cpu = argv[7];
	Classifier classifier(model_file, trained_file, mean_file, label_file, gpu_cpu);

	double pred_threshold;
	double diff_threshold;
 	int batch_size;
 	int output_number;

	/* set threshold */

	if(std::strcmp(argv[13], "output_number") == 0){
		output_number = atof(argv[14]);
		classifier.set_output_number(output_number);
	}

	if(std::strcmp(argv[8], "threshold") == 0){
		pred_threshold = atof(argv[9]);
		diff_threshold = atof(argv[10]);
		classifier.set_threshold(pred_threshold, diff_threshold);
	}
	if(std::strcmp(argv[11], "batch") == 0){
		batch_size = atoi(argv[12]);
		classifier.set_batch_size(batch_size);
	}

	if(std::strcmp(argv[7], "CPU") == 0){
		classifier.set_cpu(1);
	}

	if(std::strcmp(argv[5], "file") == 0){

		string file = argv[6];
		std::cout << "---------- Prediction for "
				<< file << " ----------" << std::endl;

		cv::Mat img = cv::imread(file, -1);
		CHECK(!img.empty()) << "Unable to decode image " << file;
		//cv::resize(img, img, cv::Size(32,100));
		std::vector<Prediction> predictions = classifier.Classify(img);

		/* Print the top N predictions. */
		for (size_t i = 0; i < predictions.size(); ++i) {
			Prediction p = predictions[i];
			std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
					<< p.first << "\"" << std::endl;
		}

	}else if(std::strcmp(argv[5], "folder") == 0){

		string image_folder = argv[6];

		vector<string> frames_to_process;
		vector<string> frames_name;
		read_frames(image_folder, frames_to_process, frames_name);

		for(int f = 0; f < frames_to_process.size(); f++){

			std::cout << "---------- Prediction for "
					<< frames_to_process[f] << " ----------" << std::endl;
			unsigned long t_init = GetTickCount();

			cv::Mat img = cv::imread(frames_to_process[f], -1);

			CHECK(!img.empty()) << "Unable to decode image " << frames_to_process[f];
			std::vector<Prediction> predictions = classifier.Classify(img);

			/* Print the top N predictions. */
			for (size_t i = 0; i < predictions.size(); ++i) {
				Prediction p = predictions[i];
				std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
						<< p.first << "\"" << std::endl;
			
			}
			unsigned long t_last = GetTickCount();
	std::cout << "classifying time: " << t_last-t_init<< "ms" << std::endl;
		}

	}else if(std::strcmp(argv[5], "lcm") == 0){

		string image_channel = argv[6];

		lcm_t* lcm = lcm_create(NULL);
		classifier.set_lcm(lcm, image_channel);
		
		std::cout << "---------- Prediction for "
				<< "lcm input channel " << image_channel << " ----------" << std::endl;
		
		classifier.run();
		
	}


}

