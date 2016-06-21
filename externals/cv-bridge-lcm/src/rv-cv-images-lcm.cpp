/**
 * @brief publish images or video stream to LCM 
 * @author: Nick Wang
 *
 */

#include <signal.h>
#include <getopt.h>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <cv_bridge_lcm/rv-cv-bridge-lcm.h>

using namespace std;
using namespace cv;

//#define SHOW_GUI

cvBridgeLCM* cv_bridge;
cv::Mat* _image_vis;
bool _finish;

void read_video_publish(int use_video_device, char* video_path, char* out_chan,
		int sleep_ms){

	string str = string(video_path);
	cv::VideoCapture cap(str);

	if (use_video_device >= 0) {
		// directly connect to a video device
		cap = cv::VideoCapture(use_video_device);

		if(!cap.isOpened()) {
			cerr << "ERROR: Can't find video device " << use_video_device << "\n";
			return;
		}

	} else {

		if(!cap.isOpened()) {
			cerr << "ERROR: Can't open video file: " << video_path << "\n";
			cerr << "Try: -f <path>" << endl;
			return;
		}
	}



	if (cap.get(CV_CAP_PROP_FRAME_WIDTH) < 640) {
		// some cams provide a small image by default, try to get a larger one
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	}

	int width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	// only allocate image_vis once
	if (_image_vis == NULL) {
		_image_vis = new cv::Mat(height, width, CV_8UC3);
	}

	while(!_finish) {

		Mat im = cv::Mat(height, width, CV_8UC3);
		cap >> im;
		cvtColor(im, im, CV_RGB2BGR);
		cv_bridge->publish_rgb(im.clone(), out_chan);

		cap >> *_image_vis;
		cv::imshow("video", *_image_vis);

		usleep(sleep_ms);
	}

	cap.release();

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
						|| it->path().extension().string() == ".bmp")
		){

			frames_to_process.push_back(it->path().string());
			frames_names.push_back(it->path().stem().string());
		}
	}
	// sort the image files
	sort(frames_to_process.begin(), frames_to_process.end());

}


void termination_handler(int signum) {

}

void setup_signal_handlers(void (*handler)(int)) {
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

static void
usage(const char *name)
{
	fprintf(
			stderr,
			"usage: %s [options]\n"
			"\n"
			"  -h, --help                   Shows this help text and exits\n"
			"  -i, --images-path <path>     images folder\n"
			"  -d, --device <index>         use video device (typical 0)\n"
			"  -v, --video-file <file>      video file\n"
			"  -t, --publish-type <int>     0: image, 1: depth\n"
			"  -o, --output-channel <chan>  output channel\n"
			"  -s, --sleep-ms               sleep micro-second default: 33000\n"
			"\n",
			name);

	exit(1);
}

int main(int argc, char** argv) {
	int use_video_device = -1; // use LCM if < 0
	char* images_folder = (char *)"";
	vector<string> frames_to_process;
	vector<string> frames_name;
	char* video_file = (char *)"";
	int publish_type = 0;
	char* chan = (char *)"IMAGE";
	long sleep_ms = 33000;

	lcm_t* lcm = lcm_create(NULL);
	cv_bridge = new cvBridgeLCM(lcm, lcm);

	const char *optstring = "hd:i:v:o:s:t:";
	int c;
	struct option long_opts[] =
	{
			{ "help", no_argument, 0, 'h' },
			{ "device", required_argument, 0, 'd' },
			{ "images-path", required_argument, 0, 'i' },
			{ "video-path", required_argument, 0, 'v' },
			{ "publish-type", required_argument, 0, 't' },
			{ "output-channel", required_argument, 0, 'o' },
			{ "sleep-ms", required_argument, 0, 's' },
			{ 0, 0, 0, 0 }
	};

	while ((c = getopt_long(argc, argv, optstring, long_opts, 0)) >= 0) {
		switch (c){
		case 'o':
			chan = strdup(optarg);
			break;
		case 'd':
			use_video_device = atoi(optarg);
			break;
		case 'i':
			images_folder = strdup(optarg);
			break;
		case 't':
			publish_type = atoi(optarg);
			break;
		case 'v':
			video_file = strdup(optarg);
			break;
		case 's':
			sleep_ms = atol(optarg);
			break;
		case 'h':
			usage(argv[0]);
			break;
		}
	}

	#ifdef SHOW_GUI
		cvNamedWindow("video");
		cvStartWindowThread();
	#endif

	if(string(images_folder).length() > 0){

		std::string image_folder = images_folder;

		read_frames(image_folder, frames_to_process, frames_name);

		cout << "there are " << frames_to_process.size() << " to process" << endl;

		for(int f = 0; f < frames_to_process.size(); f++){
			Mat im = imread(frames_to_process[f]);

			if(publish_type == 0){
				cv_bridge->publish_rgb(im.clone(), chan);
			}else if(publish_type == 1){
				Mat im_depth = im;

				cv::Mat im_kinect_rgb = Mat::zeros(480, 640, CV_8UC3);
				cv::Mat im_kinect_depth = Mat::zeros(480, 640, CV_16UC1);

				im_kinect_rgb.setTo(Scalar(0, 0, 255));

        		// [y, x]: [240, 320] -> [480, 640]
        		for(int i = 0; i < im.cols; i++){
        			for(int j = 0; j < im.rows; j++){
        				int x = i*2;
        				int y = j*2;
        				int16_t depth = im.at<Vec3b>(j, i)[0] + 255 * im.at<Vec3b>(j, i)[1];
        				//cout << x << ", " << y << ": " << depth << endl;
        				im_kinect_depth.at<int16_t>(y, x) = depth;
        				im_kinect_depth.at<int16_t>(y+1, x) = depth;
        				im_kinect_depth.at<int16_t>(y, x+1) = depth;
        				im_kinect_depth.at<int16_t>(y+1, x+1) = depth;
        			}
        		}

        		cv_bridge->publish_kinect_frame(im_kinect_rgb, im_kinect_depth, "KINECT_FRAME");

//				#ifdef SHOW_GUI
//					cv::imshow("video", im_kinect_depth);
//				#endif
			}




			usleep(sleep_ms); // 30fps
		}
	}

	if(string(video_file).length() > 0){
		read_video_publish(use_video_device, video_file, chan, sleep_ms);
	}

	#ifdef SHOW_GUI
		cvDestroyWindow("video");
	#endif

	delete cv_bridge;

	return 0;
}
