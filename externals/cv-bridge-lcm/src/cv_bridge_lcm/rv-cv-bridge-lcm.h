/*
 * cvBridgeLCM.h
 *
 *  Created on: Dec 12, 2013
 *      Author: drc
 */

#ifndef TLOPENCVLCMPUBLISHER_H_
#define TLOPENCVLCMPUBLISHER_H_

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <signal.h>
#include <fstream>
#include <iostream>

#include <lcm/lcm-cpp.hpp>
#include <lcm/lcm.h>
#include <bot_core/bot_core.h>
#include <bot_param/param_client.h>
#include <lcmtypes/bot2_param.h>
#include <lcmtypes/kinect_depth_msg_t.h>
#include <lcmtypes/kinect_image_msg_t.h>
#include <lcmtypes/kinect_frame_msg_t.h>
#include <lcmtypes/bot_core_image_t.h>
#include <lcmtypes/bot_core/image_t.hpp>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include <jpeg-utils/jpeg-utils-ijg.h>
#include <zlib.h>

class cvBridgeLCM {
public:
	cvBridgeLCM();
	cvBridgeLCM(lcm_t* publish_lcm,lcm_t* subscribe_lcm);

	virtual ~cvBridgeLCM();

	void publish_rgb(cv::Mat im, char * out_channel);

	void publish_mjpg(cv::Mat im, char * out_channel);
	void publish_mjpg(std::vector<unsigned char> compressed_buf,
			int width, int height, char * out_channel);

	void publish_gray(cv::Mat im, char * out_channel);
	void publish_gray_float(cv::Mat im, char * out_channel);
	void publish_gray_z(cv::Mat im, char * out_channel);

	void publish_depth(cv::Mat im_depth, char * channel);
	void publish_kinect_frame(cv::Mat im_rgb, cv::Mat im_depth, char * channel);
	void publish_tango_frame(char * channel);

	void setJpegQuality(int jpegQuality) {
		jpeg_quality = jpegQuality;
	}

	void setPublishLcm(lcm_t* publishLcm) {
		publish_lcm_ = publishLcm;
	}

private:
	lcm_t* publish_lcm_;
	lcm_t* subscribe_lcm_;

	int jpeg_quality;

};

#endif /* TLOPENCVLCMPUBLISHER_H_ */
