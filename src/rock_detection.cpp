#include <ros/ros.h>


#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>

// rock detection messages
#include <rock_publisher/imgData.h>
#include <rock_publisher/imgDataArray.h>

namespace enc = sensor_msgs::image_encodings;

static const char WINDOW[] = "Image window";

class RockDetection
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher detect_pub_;

  // Reusable images
  // convert to HSV
  cv::Mat hsvImg;
  // threshold based upon hue channel
  cv::Mat coldMask;
  cv::Mat warmMask;
  cv::Mat mask;
  cv::Mat mask2;
  // saturation masks
  cv::Mat satMask;
  cv::Mat highSatMask;

  // calibration things
  std::vector<cv::Scalar> mins; // these vecs each follow the same ordering
  std::vector<cv::Scalar> maxs;
  std::vector<cv::Scalar> rgbAvgs;
 
public:
  RockDetection()
    : it_(nh_)
  {
    //image_pub_ = it_.advertise("image_detect", 1);
    image_sub_ = it_.subscribe("image", 1, &RockDetection::imageCb, this);
    //image_sub_ = it_.subscribe("image_raw", 1, &RockDetection::imageCb, this);
    detect_pub_ = nh_.advertise<rock_publisher::imgDataArray>( "detects", 1000 ) ;

    // getCalibrations() will load up a vector of scalars of min hsv values,
    // 'mins', a vector of scalars of max hsv values, 'max,' 
    // and a vector of scalars of rgb values, 'rgbAvgs,' 
    // representing the avg rgb value of each min-max pair.
   // getCalibrations(); // TODO: problem with file load?
 
  }

  ~RockDetection()
  {
    cv::destroyWindow(WINDOW);
  }

  void getCalibrations()
  { 
    // yaml parsing begins
    //std::ifstream fin("../sunny.yml", std::ifstream::in);
    std::ifstream fin("/cloudy.yml", std::ifstream::in);
    YAML::Parser parser(fin);
    YAML::Node calibrationOutput;
    parser.GetNextDocument(calibrationOutput);

    getHsvMinsAndMaxesFromYaml(calibrationOutput);

    calcRgbAveragesForBoxColors();
  } 

  void getHsvMinsAndMaxesFromYaml(const YAML::Node& cal)
  {
    for (unsigned i = 0; i < cal["colors"].size(); ++i)
    { // for each color
      int h, s, v;
      cal["colors"][i]["mins"]["h"] >> h;
      cal["colors"][i]["mins"]["s"] >> s;
      cal["colors"][i]["mins"]["v"] >> v;
      mins.push_back(cv::Scalar(h, s, v));

      cal["colors"][i]["maxs"]["h"] >> h;
      cal["colors"][i]["maxs"]["s"] >> s;
      cal["colors"][i]["maxs"]["v"] >> v;
      maxs.push_back(cv::Scalar(h, s, v));
    }
  }

  void calcRgbAveragesForBoxColors()
  {
    for (unsigned i = 0; i < mins.size(); ++i)
    {
      cv::Mat min(1, 1, CV_8UC3, mins.at(i));
      cvtColor(min, min, CV_HSV2RGB);
      cv::Mat max(1, 1, CV_8UC3, maxs.at(i));
      cvtColor(max, max, CV_HSV2RGB);
      cv::Mat avg = min + max / 2;
      // std::cout << (int)avg.at<cv::Vec3b>(0,0)[0] << ", " <<
      //              (int)avg.at<cv::Vec3b>(0,0)[1] << ", " <<
      //              (int)avg.at<cv::Vec3b>(0,0)[2] << std::endl;

      // casting chars being used as ints to ints
      rgbAvgs.push_back(cv::Scalar((int)avg.at<cv::Vec3b>(0,0)[0],
                                   (int)avg.at<cv::Vec3b>(0,0)[1],
                                   (int)avg.at<cv::Vec3b>(0,0)[2]));
    }
    std::cout << std::endl;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }


    // do image processing here
    // OpenCV Mat image is cv_ptr->image

    // convert to HSV
    // reduce image size (faster processing, less detail) and apply Gauss blur
    cv::pyrDown(cv_ptr->image, hsvImg);

    cvtColor(hsvImg, hsvImg, CV_BGR2HSV,  CV_8U);
    //cvtColor(cv_ptr->image, hsvImg, CV_BGR2HSV,  CV_8U);

    // split channels
    std::vector<cv::Mat> hsv;
    split(hsvImg, hsv);
   
    // threshold based upon hue channel
    // define colors as ranges or with a color radius
    // H: 0 - 180, S: 0 - 255 
    int radius = 15; 
    static const int green = 60;
    static const int purple = 135;
    static const int lightBlue = 100;
    static const int yellow = 30;
    static const int orange = 15 ;
    
    inRange(hsv[0], green-radius, green+radius, mask2);
    coldMask = mask2.clone();
    inRange(hsv[0], purple-radius, purple+radius, mask2);
    coldMask |= mask2; // want XOR
    inRange(hsv[0], lightBlue-radius, lightBlue+radius, mask2);
    coldMask |= mask2;

    inRange(hsv[0], orange-radius, orange+radius, warmMask);
    inRange(hsv[0], yellow-radius, yellow+radius, mask2);
    warmMask |= mask2;

    // threshold on saturation channel (get rid of low saturation pixels:
    // white/brown)
    inRange(hsv[1], 50, 255,satMask); // yay!!
    //inRange(hsv[1], 100, 255, highSatMask); // yay!!
    inRange(hsv[1], 50, 255, highSatMask); // yay!!
    
    // AND mask with saturation channel
    coldMask &= satMask; // works OK  with current color calibration 
    //warmMask &= satMask; // works OK  with current color calibration 
    warmMask &= highSatMask; // works OK  with current color calibration 
   
    // apply morphological operations to get rid of noise
    static int dilationElem = cv::MORPH_RECT; // create me once!
    //static cv::Mat structureElem = getStructuringElement(dilationElem, cv::Size(9,9)); 
    static cv::Mat structureElem = getStructuringElement(dilationElem, cv::Size(5,5)); // create me once!
    //morphologyEx(warmMask, warmMask, cv::MORPH_OPEN, structureElem);
    erode(warmMask, warmMask, structureElem);

    // merge different masks
    mask = coldMask + warmMask;
    
    // use findContours to filter and blob detected rocks
    //std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > warmContours;
    std::vector<std::vector<cv::Point> > coldContours;
    std::vector<cv::Vec4i> hierarchy;

    // running this on mask seems to alter it
    //findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); 
    //findContours(satMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); 
    // for close range targets:
    //findContours(coldMask.clone(), coldContours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); 
    //findContours(warmMask.clone(), warmContours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); 
    findContours(coldMask.clone(), coldContours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1); 
    findContours(warmMask.clone(), warmContours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1); 
 
    // dynamic? 
    //int numContours = contours.size();
    // far away : static const int MAX_ROCK_SIZE = 800;
    // static const int MIN_ROCK_SIZE = 20;
    static const int MAX_ROCK_SIZE = 1300;
    static const int MIN_ROCK_SIZE = 100;
    static const int MAX_COMPACT_NUM = 3.5; // 1 is perfect circle
    // rock bounding boxes
    std::vector<cv::Rect> detections;  
    // contour area
    double area(0);
    // contour length 
    double length(0);
    // blob compactness 
    double compactness(0);
    // blob compactness 
    double entropy(0);
    // number of mask types (warm, cold)
    int numMaskTypes = 1; 
    
    int numContours = 0;

    //printf("%d coldContours\n", coldContours.size() );
    //printf("%d warmContours\n", warmContours.size());
	
    coldContours.insert( coldContours.end(), warmContours.begin(), warmContours.end() );
    numContours = coldContours.size();

    // Detection data structures for publishing
    rock_publisher::imgData rockData ;
    rock_publisher::imgDataArray rocksMsg ;

    for(int i; i< numContours; ++i)
    {
	 // a contour is an array points
	 length = arcLength(coldContours[i], true);
	 area = contourArea(coldContours[i]);
	 compactness = (length*length) / (4 * 3.14159 * area);
	 // calculate entropy on detection ROI
	 cv::Rect rect =  boundingRect(coldContours[i]);
	 cv::Mat roi = hsv[0](rect);
	 cv::Scalar mn = 0, stdev = 0;
	 meanStdDev(roi, mn, stdev);	
	 // width-to-height ratios
	 float hToW = rect.height/rect.width;	  
	 float wToH =  rect.width/rect.height;	  
	

	 if( area < MAX_ROCK_SIZE && area > MIN_ROCK_SIZE )
	 {	
		//printf("contour %d: a = %f l = %f [compact = %f]\n", i, area, length, compactness);
		//printf("contour %d: mn = %f stdev = %f\n",i, mn.val[0], stdev.val[0]);

		// draw rectangle
		//if( hToW > 0.20 || wToH > 0.20 )
		if( compactness < MAX_COMPACT_NUM )
         	{
			printf("contour %d: mn = %f stdev = %f\n",i, mn.val[0], stdev.val[0]);
			printf("contour %d: a = %f l = %f [compact = %f]\n", i, area, length, compactness);
			cv::rectangle(cv_ptr->image, 2*rect.tl(), 2*rect.br(),  cv::Scalar(0,255,0), 2);
			//cv::drawContours(cv_ptr->image, coldContours, i, cv::Scalar(0,0,255), 2); 

			// ----------- add detections to array of detections -----------------
			rockData.x = rect.x ;
			rockData.y =  rect.y ;
			rockData.width = rect.width ;
			rockData.height =  rect.height ;
			//rockData.color = "rainbow" ;
			rocksMsg.rockData.push_back(rockData) ;

			// -------------------------------------------------------------------
         	}
	 }

    }

    // only for visualization	    
    // merge results
    merge(hsv, hsvImg);

    // convert back to RGB
    static cv::Mat rgbImg; 
    cvtColor(hsvImg, rgbImg , CV_HSV2BGR,  CV_8U);

    // just show colored rocks (RGB mask)
    std::vector<cv::Mat> rgb;
    rgb.push_back(mask);
    rgb.push_back(mask);
    rgb.push_back(mask);
    cv::Mat rgbMask;
    merge(rgb, rgbMask);
    rgbImg &= rgbMask;

    // resize image to original scale

    pyrUp(rgbImg, rgbImg);

    //cv::imshow("mask", mask);
    //cv::imshow("mask rgb", rgbImg);
    //cv::imshow("small img rgb", imgSmall);
    //cv::imshow("sat", satMask);
    //cv::imshow("value", hsv[2]);
    //cv::imshow(WINDOW, hsvImg);
    cv::imshow("detections!", cv_ptr->image);
    cv::waitKey(3);

    // publish images    
    ////image_pub_.publish(cv_ptr->toImageMsg());
    // publish detections
    detect_pub_.publish(rocksMsg) ;
  }


};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rover_cam_detect");
  RockDetection rd;
  ros::spin();
  return 0;
}
