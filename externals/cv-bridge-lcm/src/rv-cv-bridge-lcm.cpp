/*
 * cvBridgeLCM.cpp
 *
 *  Created on: Dec 12, 2013
 *      Author: drc
 */

#include "cv_bridge_lcm/rv-cv-bridge-lcm.h"

using namespace std;
using namespace cv;

cvBridgeLCM::cvBridgeLCM() {
	// TODO Auto-generated constructor stub

	this->jpeg_quality = 94;
}

cvBridgeLCM::cvBridgeLCM(lcm_t* publish_lcm,lcm_t* subscribe_lcm):
          publish_lcm_(publish_lcm), subscribe_lcm_(subscribe_lcm) {

	this->jpeg_quality = 94;
}

cvBridgeLCM::~cvBridgeLCM() {
	// TODO Auto-generated destructor stub
}

void cvBridgeLCM::publish_rgb(Mat im, char * out_channel){

	bot_core_image_t msg_rgb; // RGB
	msg_rgb.utime = bot_timestamp_now();
	msg_rgb.width = im.cols;
	msg_rgb.height = im.rows;
	msg_rgb.row_stride = 3 * im.cols;
	msg_rgb.pixelformat = BOT_CORE_IMAGE_T_PIXEL_FORMAT_RGB;
	msg_rgb.size = im.cols * im.cols *3;
	msg_rgb.nmetadata=0;
	msg_rgb.metadata=NULL;

	msg_rgb.data = im.data;

	bot_core_image_t_publish(publish_lcm_, out_channel, &msg_rgb);
}

void cvBridgeLCM::publish_gray(Mat im, char * out_channel){
	bot_core_image_t msg_gray; // GRAY
	msg_gray.utime = bot_timestamp_now();
	msg_gray.width = im.cols;
	msg_gray.height = im.rows;
	msg_gray.row_stride = im.cols;
	msg_gray.pixelformat = BOT_CORE_IMAGE_T_PIXEL_FORMAT_GRAY;
	msg_gray.size = im.cols * im.rows;
	msg_gray.nmetadata=0;
	msg_gray.metadata=NULL;

	msg_gray.data = im.data;
	bot_core_image_t_publish(publish_lcm_, out_channel, &msg_gray);
}

void cvBridgeLCM::publish_gray_float(Mat im, char * out_channel){
	bot_core_image_t msg_gray; // GRAY
	msg_gray.utime = bot_timestamp_now();
	msg_gray.width = im.cols;
	msg_gray.height = im.rows;
	msg_gray.row_stride = im.cols;
	msg_gray.pixelformat = BOT_CORE_IMAGE_T_PIXEL_FORMAT_FLOAT_GRAY32;
	msg_gray.size = im.cols * im.rows;
	msg_gray.nmetadata=0;
	msg_gray.metadata=NULL;

	msg_gray.data = im.data;
	bot_core_image_t_publish(publish_lcm_, out_channel, &msg_gray);
}

void cvBridgeLCM::publish_mjpg(Mat im, char * out_channel){

	bot_core_image_t msg_rgb_jpg; // RGB jpeg
	msg_rgb_jpg.utime = bot_timestamp_now();
	msg_rgb_jpg.width = im.cols;
	msg_rgb_jpg.height = im.rows;
	msg_rgb_jpg.row_stride = 3 * im.cols;
	msg_rgb_jpg.pixelformat = BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG;
	msg_rgb_jpg.nmetadata=0;
	msg_rgb_jpg.metadata=NULL;

	// do compression
	std::vector<int> params;
	params.push_back(cv::IMWRITE_JPEG_QUALITY);
	params.push_back( this->jpeg_quality);

	vector<unsigned char> buf;
	cv::cvtColor( im, im, CV_BGR2RGB);
	if (!cv::imencode(".jpg", im, buf, params)) {
		cout << "Error encoding jpeg image" << endl;
	}
	uint8_t* buf2 = new uint8_t[buf.size()];
	for(int i = 0; i < buf.size(); i++){
		buf2[i] = (uint8_t)buf[i];
	}
	msg_rgb_jpg.data = buf2;
	msg_rgb_jpg.size = buf.size() * sizeof(uint8_t);

	bot_core_image_t_publish(publish_lcm_, out_channel, &msg_rgb_jpg);

	free(buf2);
}

void cvBridgeLCM::publish_mjpg(vector<unsigned char> buf, int width, int height,
		char * out_channel){

	bot_core_image_t msg_rgb_jpg; // RGB jpeg
	msg_rgb_jpg.utime = bot_timestamp_now();
	msg_rgb_jpg.width = width;
	msg_rgb_jpg.height = height;
	msg_rgb_jpg.row_stride = 3 * width;
	msg_rgb_jpg.pixelformat = BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG;
	msg_rgb_jpg.nmetadata=0;
	msg_rgb_jpg.metadata=NULL;

	uint8_t* buf2 = new uint8_t[buf.size()];
	for(int i = 0; i < buf.size(); i++){
		buf2[i] = (uint8_t)buf[i];
	}
	msg_rgb_jpg.data = buf2;
	msg_rgb_jpg.size = buf.size() * sizeof(uint8_t);

	bot_core_image_t_publish(publish_lcm_, out_channel, &msg_rgb_jpg);

	free(buf2);
}

void cvBridgeLCM::publish_gray_z(Mat im, char * out_channel){
	bot_core_image_t msg_gray; // GRAY
	msg_gray.utime = bot_timestamp_now();
	msg_gray.width = im.cols;
	msg_gray.height = im.rows;
	msg_gray.row_stride = im.cols;
	msg_gray.pixelformat = BOT_CORE_IMAGE_T_PIXEL_FORMAT_GRAY;
	msg_gray.nmetadata=0;
	msg_gray.metadata=NULL;

	// do compression
	int uncompressed_size = im.cols * im.rows * sizeof(int8_t);
	cout << sizeof(int8_t) << ", " << sizeof(short) << endl;
	// allocate space for zlib compressing depth data
	unsigned long compress_buf_size = im.cols * im.rows * sizeof(int16_t) * 4;
	uint8_t* compress_buf = (uint8_t*) malloc(compress_buf_size);

	compress2( compress_buf, &compress_buf_size, (const Bytef*) im.data, uncompressed_size,
	                  Z_BEST_SPEED);

	msg_gray.data = compress_buf;
	msg_gray.size = (int)compress_buf_size;

	bot_core_image_t_publish(publish_lcm_, out_channel, &msg_gray);
}

void cvBridgeLCM::publish_depth(cv::Mat im_depth,
		char* channel) {

	kinect_depth_msg_t msg;

	int64_t thisTime = bot_timestamp_now();
	msg.timestamp = thisTime;

    msg.timestamp = thisTime;
    msg.width = 320;
    msg.height = 160;
    msg.depth_data_format = KINECT_DEPTH_MSG_T_DEPTH_MM;//KINECT_DEPTH_MSG_T_DEPTH_11BIT;
    msg.depth_data_nbytes = 320 * 160 * sizeof(short);
    msg.compression = KINECT_DEPTH_MSG_T_COMPRESSION_NONE;
    msg.uncompressed_size = msg.depth_data_nbytes;
    msg.depth_data = im_depth.data;

    kinect_depth_msg_t_publish(publish_lcm_, "IMAGE_DPETH", &msg);
}

void cvBridgeLCM::publish_kinect_frame(cv::Mat im_rgb, cv::Mat im_depth,
		char* channel) {

	kinect_frame_msg_t msg;

	int64_t thisTime = bot_timestamp_now();
	msg.timestamp = thisTime;

	msg.image.timestamp = thisTime;
	msg.image.width = im_rgb.cols;
	msg.image.height = im_rgb.rows;
    msg.image.image_data_format = KINECT_IMAGE_MSG_T_VIDEO_RGB;
    msg.image.image_data = im_rgb.data;
    msg.image.image_data_nbytes = im_rgb.cols * im_rgb.rows * 3;

    msg.depth.timestamp = thisTime;
    msg.depth.width = im_depth.cols;
    msg.depth.height = im_depth.rows;
    msg.depth.depth_data_format = KINECT_DEPTH_MSG_T_DEPTH_MM;//KINECT_DEPTH_MSG_T_DEPTH_11BIT;
    msg.depth.depth_data_nbytes = im_depth.cols * im_depth.rows * sizeof(short);
    msg.depth.compression = KINECT_DEPTH_MSG_T_COMPRESSION_NONE;
    msg.depth.uncompressed_size = msg.depth.depth_data_nbytes;
    msg.depth.depth_data = im_depth.data;

    kinect_frame_msg_t_publish(publish_lcm_, channel, &msg);
}


