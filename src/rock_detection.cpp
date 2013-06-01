#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/ColorRGBA.h> 
#include <std_msgs/Int32.h> 
#include <std_msgs/String.h> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>

// rock detection messages
#include <rock_publisher/imgData.h>
#include <rock_publisher/imgDataArray.h>

//recalibration message
#include <rock_publisher/recalibrateMsg.h>

namespace enc = sensor_msgs::image_encodings;

static const char WINDOW[] = "Image window";

// Enumerate different rock colors (subject to change
// based upon UI input)
// b =0, g = 1, r = 2, o = 3, p = 4, y = 5
enum colors_ { blue_ = 0, green_, red_,
               orange_, purple_, yellow_, numColors_ };

class RockDetection
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher detect_pub_;
  // detector parameters
  ros::Subscriber saturation_sub_; // get adjustment from UI slider
  ros::Publisher saturation_pub_; // get adjustment from UI slider

  ros::Subscriber recalibrate_sub_; //get recalibration messages

  // Reusable images
  // convert to HSV
  cv::Mat hsvImg;
  // threshold based upon hue channel
  cv::Mat mask;
  cv::Mat mask2;
  // saturation masks
  cv::Mat satMask;
  cv::Mat highSatMask;
  // bounding area mask (e.g. only detect things in these areas
  cv::Mat boundMask;
 
  // color calibration:
  // use a vector of vectors as a calibration 'stack' 
  // to accomondate on the fly calibration
  std::vector< std::vector<cv::Scalar> > mins; // these vecs each follow the same ordering
  std::vector<std::vector<cv::Scalar> > maxs;
  
  std::vector<std_msgs::ColorRGBA> rgbaAvgs;

  // config parameters 
  int minSaturation_;
  // range of calibration thresholds - based upon mean +- #(std devs) 
  // from ROI rock patch
  int NUM_CALIB_STD_DEVS; 
  int MAX_ROCK_SIZE;
  int MIN_ROCK_SIZE;
  int UL_X;
  int UL_Y;
  int LR_X;
  int LR_Y;
  double MAX_COMPACT_NUM; // MAX_COMPACT_NUM = 1, for a perfect circle
  std::string PATH_TO_CALIBRATIONS;
  std::string CALIBRATION_FILE;
  bool SHOW_VIZ; // show debug visualizations

public:
  RockDetection()
    : it_(nh_), minSaturation_(50), MAX_ROCK_SIZE(100), MIN_ROCK_SIZE(20), MAX_COMPACT_NUM(3.0), 
	NUM_CALIB_STD_DEVS(2.0), UL_X(0), UL_Y(0), LR_X(864), LR_Y(480), SHOW_VIZ(false)
  {

    // for visualization
    if(SHOW_VIZ) 
     cvNamedWindow(WINDOW);
 
    bool latch = true; 
    image_pub_ = it_.advertise("image_detect", 1);
    image_sub_ = it_.subscribe("image", 1, &RockDetection::imageCb, this);
    detect_pub_ = nh_.advertise<rock_publisher::imgDataArray>( "detects", 1000 ) ;
    // parameter pub/sub
    saturation_sub_ = nh_.subscribe("detect_saturation", 1, &RockDetection::saturationCb, this);
    saturation_pub_ = nh_.advertise<std_msgs::Int32>("detect_saturation_info", 1, latch);
    //Start subscriber to recalibrate topic, which calls the recalibrate function
    recalibrate_sub_ = nh_.subscribe("recalibrate", 1000, &RockDetection::recalibrateCallback,  this);
    // (1) load detection parameters
    // (2) getCalibrations() will load up a vector of scalars of min hsv values,
    // 'mins', a vector of scalars of max hsv values, 'max,' 
    // and a vector of scalars of rgba values, 'rgbaAvgs,' 
    // representing the avg rgb value of each min-max pair


    bool status = false;
    if( (status = getDetectionParams()) )
    {
      status = getCalibrations(); 
    }

    if(!status)
    {
      ROS_ERROR("Could not load calibration or parameters file. Shutting down." );
      nh_.shutdown();
      return;
    }
 
  }

  ~RockDetection()
  {
    cv::destroyWindow(WINDOW);
  }

  bool getDetectionParams()
  {
    if (nh_.hasParam("rover_cam_detect/maxRockSize") && // default = 100
        nh_.hasParam("rover_cam_detect/minRockSize") && // default = 15 
        nh_.hasParam("rover_cam_detect/minX") && // default = 15 
        nh_.hasParam("rover_cam_detect/minY") && // default = 15 
        nh_.hasParam("rover_cam_detect/maxX") && // default = 15 
        nh_.hasParam("rover_cam_detect/maxY") && // default = 15 
        nh_.hasParam("rover_cam_detect/showViz") && // default = 15 
        nh_.hasParam("rover_cam_detect/maxCompactNum") && // default = 3.0
        nh_.hasParam("rover_cam_detect/calibPath") && // default = /home/csrobot/.calibrations/
        nh_.hasParam("rover_cam_detect/calibFile") ) // default = /default_calib.yml
    {

      nh_.getParam("rover_cam_detect/maxRockSize", MAX_ROCK_SIZE);
      nh_.getParam("rover_cam_detect/minRockSize", MIN_ROCK_SIZE);
      nh_.getParam("rover_cam_detect/maxCompactNum", MAX_COMPACT_NUM);
      nh_.getParam("rover_cam_detect/calibPath", PATH_TO_CALIBRATIONS);
      nh_.getParam("rover_cam_detect/calibFile", CALIBRATION_FILE);
      nh_.getParam("rover_cam_detect/minX", UL_X );
      nh_.getParam("rover_cam_detect/minY", UL_Y);
      nh_.getParam("rover_cam_detect/maxX", LR_X); 
      nh_.getParam("rover_cam_detect/maxY", LR_Y);
      nh_.getParam("rover_cam_detect/showViz", SHOW_VIZ);

      ROS_INFO("max rock size: %d\n", MAX_ROCK_SIZE);
      ROS_INFO("min rock size: %d\n", MIN_ROCK_SIZE);
      ROS_INFO("max compact number: %f\n", MAX_COMPACT_NUM);
      ROS_INFO("minX: %d minY: %d maxX: %d maxY: %d", UL_X, UL_Y, LR_X, LR_Y);
      ROS_INFO("visualization turned on : %d\n", SHOW_VIZ);
 
/*     ROS_INFO("calibration file directory: %s\n", PATH_TO_CALIBRATIONS.c_str());
      ROS_INFO("calibration file: %s\n", CALIBRATION_FILE.c_str());
*/
      ROS_INFO("Rock detections parameters loaded!");
    } 
    else {
      ROS_ERROR("some detection params don't exist: are you having namespace issues? ");
      return false;
    }
  
    return true;

  }


  bool getCalibrations()
  { 
    // yaml parsing begins
    std::string calibrationPath = PATH_TO_CALIBRATIONS + CALIBRATION_FILE;
    ROS_INFO("Loading calibration file: %s\n", calibrationPath.c_str());

    std::ifstream fin(calibrationPath.c_str(), std::ifstream::in);
    
    if (!fin)
    {
      ROS_ERROR("There is no color calibration file at: %s\n", 
                calibrationPath.c_str());
      ROS_ERROR("HEY!! DID YOU COPY default_calib.yml (in /rover_cam_detect) info the .calibrations directory? Do this if you for test and/or haven't already done another calibration and changed the parameter setting in the launch file...."); 
      return false; 
    }

    try {
       YAML::Parser parser(fin);
       YAML::Node calibrationOutput;
       parser.GetNextDocument(calibrationOutput);

       initCalibrationVectors();

       getHsvMinsAndMaxesFromYaml(calibrationOutput);

       //calcRgbAveragesForBoxColors();
     }
     catch(YAML::ParserException& e) {
       std::cout << e.what() << "\n";
       return false;
     }
    
     return true; 
  } 

  void initCalibrationVectors()
  {
 
      // initialize calibration vectors to 0
      std::vector<cv::Scalar> vec(1,cv::Scalar(-1,-1,-1));
  
      // add vectors (containing thresholds) for each color 
      // this vector correspons exactly to enumerated colors above
      for(unsigned i = 0; i<numColors_; ++i)
      { 
 	mins.push_back(vec);
	maxs.push_back(vec);
      }
        
  }

  // TODO: update this based upon new color identifiers
  void getHsvMinsAndMaxesFromYaml(const YAML::Node& cal)
  {
    // Read in color calibrations (thresholds) based up
    // number available (this will change when we have a fixed #)
    for (unsigned i = 0; i < cal["colors"].size(); ++i)
    { // for each color
      int h, s, v;
      cal["colors"][i]["mins"]["h"] >> h;
      cal["colors"][i]["mins"]["s"] >> s;
      cal["colors"][i]["mins"]["v"] >> v;
      mins[i][0] = cv::Scalar(h, s, v); // use vec of vecs
      //mins[i].push_back(cv::Scalar(h, s, v)); // use vec of vecs
      ROS_INFO("min hsv : %d %d %d", h, s, v);

      cal["colors"][i]["maxs"]["h"] >> h;
      cal["colors"][i]["maxs"]["s"] >> s;
      cal["colors"][i]["maxs"]["v"] >> v;
      maxs[i][0] = cv::Scalar(h, s, v); // use vec of vecs
      //maxs[i].push_back(cv::Scalar(h, s, v));
      ROS_INFO("max hsv : %d %d %d", h, s, v);
    }

    ROS_INFO("Done loading yaml calibration file.");
  }

  void calcRgbAveragesForBoxColors()
  {
    for (unsigned i = 0; i < mins.size(); ++i)
    {
      cv::Mat min(1, 1, CV_8UC3, mins[i].back() ); // use last or 'latest' calibration value
      //cv::Mat min(1, 1, CV_8UC3, mins.at(i));
      cvtColor(min, min, CV_HSV2RGB);
      cv::Mat max(1, 1, CV_8UC3, maxs[i].back() );
      //cv::Mat max(1, 1, CV_8UC3, maxs.at(i));
      cvtColor(max, max, CV_HSV2RGB);
      cv::Mat avg = min + max / 2;

      // there is no cool constructor for this :(
      std_msgs::ColorRGBA rgbaAvg;
      rgbaAvg.r = (float)avg.at<cv::Vec3b>(0,0)[0];
      rgbaAvg.g = (float)avg.at<cv::Vec3b>(0,0)[1];
      rgbaAvg.b = (float)avg.at<cv::Vec3b>(0,0)[2];
      rgbaAvg.a = 1.0; 

      rgbaAvgs.push_back(rgbaAvg);
    }

   // print statement to make sure rgba values are sane
   //
   // std::vector<std_msgs::ColorRGBA>::iterator it;
   // for (it = rgbaAvgs.begin(); it != rgbaAvgs.end(); ++it)
   // {
   //   std::cout << it->r << ", " << it->g << ", " <<
   //                it->b << ", " << it->a << std::endl;
   // }
  }

  cv::Mat createBoundAreaMask(const cv::Mat &origImg, const int minX, const int minY, 
				const int  maxX, const int maxY)
  {
    cv::Mat mask =  cv::Mat::zeros(origImg.size(), CV_8U);

    // If dimensions are bad, return a mask of one (e.g. no bounding area)   
    if( minX < 0  && minY < 0 && maxX > origImg.size().width && maxX > origImg.size().height)
    {
	ROS_INFO("Returning full detection area mask.");
	return cv::Mat::ones(origImg.size(), CV_8U); 
    }

    for(int i=minY; i<maxY; ++i)
    {
    	for(int j=minX; j<maxX; ++j)
	{
		mask.at<uchar>(i, j) = 255;
	}
    }
	
    //cv::imshow("bound mask", mask);
    //cv::waitKey(3);

    ROS_INFO("Returning some detection area mask.");
    return mask;
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

    // initialize bounding area mask
    static bool initBoundMask = false;
    if(!initBoundMask)
    {
	ROS_INFO("Created bounding area mask!");
    	boundMask = createBoundAreaMask(cv_ptr->image, UL_X, UL_Y, LR_X, LR_Y);
    	//boundMask = createBoundAreaMask(cv_ptr->image, UL_X, UL_Y, 20, 60);
	initBoundMask = true;
    }

 
    // --------- start color processing ------------ // 
    
    // reduce image size (faster processing, less detail) and apply Gauss blur
    //pyrMeanShiftFiltering(cv_ptr->image, hsvImg, 5, 15);
    //cv::pyrDown(hsvImg, hsvImg);

    cv::pyrDown(cv_ptr->image, hsvImg);
    hsvImg = equalizeIntensity(hsvImg.clone());

    // convert to HSV
    cvtColor(hsvImg, hsvImg, CV_BGR2HSV,  CV_8U);
    //cvtColor(cv_ptr->image, hsvImg, CV_BGR2HSV,  CV_8U);

    // split channels
    std::vector<cv::Mat> hsv;
    split(hsvImg, hsv);
  
    // threshold based upon hue channel
    // define colors as ranges or with a color radius
    // H: 0 - 180, S: 0 - 255 
    // approx mean hue:  green = 60 purple = 135 lightBlue = 100 yellow = 30  orange = 15 
   
    // Apply thresholds from calibration
    // mins / maxs
    cv::Mat hueMask =  cv::Mat::zeros(hsv[0].rows, hsv[0].cols, CV_8U);
    
    for(int j=0; j<numColors_; ++j) // use pre-decided set of colors size
    {
	// skip if no calibration provided for this color (init'd to -1) 
	if( ( mins[j].back() )[0] == -1 )
		continue; 

	cv::Scalar &hsvMin = mins[j].back();  // Scalar for given color
	cv::Scalar &hsvMax = maxs[j].back();
	int hMin = hsvMin[0]; // index into a Scalar
	int hMax = hsvMax[0];
	int hDiff = hMax-hMin;
	int sMin = hsvMin[1];
	int sMax = hsvMax[1];
	int sDiff = sMax-sMin;
	int vMin = hsvMin[2];
	int vMax = hsvMax[2];

	// Use hue only
    	//inRange(hsv[0], hMin+0.10*hDiff, hMax+0.1*hDiff, hueMask);
    	//inRange(hsv[0], hMin-0.10*hDiff, hMax-0.1*hDiff, hueMask);
    	//inRange(hsv[0], hMin, hMax, hueMask);
	// hue and saturation threshs - kinda sucks
    	inRange(hsv[0], hMin, hMax, mask);
    	//inRange(hsv[1], sMin, sMax, mask2);
        //mask &= mask2;
        // merge current color mask with others
        hueMask |= mask; 
	
    }
    
    // Threshold using a minimum saturation level. Low saturation
    // colors are whites/browns/grays which we don't want 
    inRange(hsv[1], minSaturation_, 255,satMask); 
    // AND mask with saturation channel
    hueMask &= satMask;  
  
    // mask out areas with bounding area mask (e.g areas of the image we want to detect on)
    //hueMask &= boundMask;
 
    //cv::imshow("hue mask", hueMask);

    // apply morphological operations to get rid of noise
    static int dilationElem = cv::MORPH_RECT; // create me once!
    //static cv::Mat structureElem = getStructuringElement(dilationElem, cv::Size(5,5)); // create me once!
    static cv::Mat structureElem = getStructuringElement(dilationElem, cv::Size(3,3)); // create me once!

    // erode image to get rid of small noisy pixels in background
    erode(hueMask, mask, structureElem);
    //dilate(mask, mask, structureElem);
     
    // --------- start shape detection processing ------------ // 
    // use findContours to filter and blob detected rocks
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    // running this on mask seems to alter it
    //findContours(mask.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); 
    //findContours(mask.clone(), contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); 
    findContours(mask.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1); 
 
    // contour area
    double area(0);
    // contour length 
    double length(0);
    // blob compactness 
    double compactness(0);
    // blob entropy
    double entropy(0);
    // processing rectangle
    cv::Rect rect;
	
    int numContours = contours.size();

    // Detection data structures for publishing
    rock_publisher::imgData rockData ;
    rock_publisher::imgDataArray rocksMsg ;

    for(int i; i< numContours; ++i)
    {
	 // a contour is an array points
	 length = arcLength(contours[i], true);
	 area = contourArea(contours[i]);
	 compactness = (length*length) / (4 * 3.14159 * area);
	 // calculate entropy on detection ROI (look for areas with low entropy)
	 // (not being used at the moment)
	 //rect =  boundingRect(contours[i]);
	 //cv::Mat roi = hsv[0](rect);
	 //cv::Scalar mn = 0, stdev = 0;
	 //meanStdDev(roi, mn, stdev);
		
	 if( area < MAX_ROCK_SIZE && area > MIN_ROCK_SIZE )
	 {	
		//printf("contour %d: a = %f l = %f [compact = %f]\n", i, area, length, compactness);
		//printf("contour %d: mn = %f stdev = %f\n",i, mn.val[0], stdev.val[0]);

		// draw rectangle
		if( compactness < MAX_COMPACT_NUM )
         	{
			// get average rbg color of contour bbox 
	 		rect =  boundingRect(contours[i]);
			static cv::Scalar rgbMean, rgbSd;
			//std_msgs::ColorRGBA rectColor = computeROIStats(cv_ptr->image, rect, rgbMean, rgbSd);
			rock_publisher::colorRGBA rectColor = computeROIStats(cv_ptr->image, rect, rgbMean, rgbSd);

			//printf("contour %d: mn = %f stdev = %f\n",i, mn.val[0], stdev.val[0]);
			//printf("contour %d: a = %f l = %f [compact = %f]\n", i, area, length, compactness);
			cv::rectangle(cv_ptr->image, 2*rect.tl(), 2*rect.br(),  cv::Scalar(0,255,0), 2);
			//cv::drawContours(cv_ptr->image, contours, i, cv::Scalar(0,0,255), 2); 

			// ----------- add detections to array of detections -----------------
			rockData.x = rect.x ;
			rockData.y =  rect.y ;
			rockData.width = rect.width ;
			rockData.height =  rect.height ;
			rockData.color = rectColor;
			rocksMsg.rockData.push_back(rockData) ;
			// -------------------------------------------------------------------
         	}
	 }
    }


   // Algorithm visualization
   if(SHOW_VIZ) 
    {
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
    	cv::imshow("sat mask", satMask);
    	cv::imshow("hue mask", hueMask);
    	cv::imshow("mask rgb", rgbImg);
    	//cv::imshow("hue", hsv[0]);
    	cv::imshow("saturation", hsv[1]);
    	//cv::imshow("value - lightness", hsv[2]);
    	cv::imshow("detections!", cv_ptr->image);
	cv::waitKey(1);
    }

    // publish detection image (with bounding boxes)   
    image_pub_.publish(cv_ptr->toImageMsg());

    // publish detections
    if(!rocksMsg.rockData.empty())
    {
       detect_pub_.publish(rocksMsg) ;
    }

  }

  // Adjust the min saturation 
  inline void saturationCb(const std_msgs::Int32::ConstPtr& msg)
  {
    if(msg->data <= 255)
    {
	minSaturation_ = msg->data;
	ROS_INFO("Saturation adjusted: %d", minSaturation_);
	saturation_pub_.publish(msg); // ECHO!
    }  
  }

  // Apply histogram equalization 
  cv::Mat equalizeIntensity(const cv::Mat& inputImage)
  {
    if(inputImage.channels() >= 3)
    {
        cv::Mat hsv;

        cvtColor(inputImage,hsv,CV_BGR2HSV);

        std::vector<cv::Mat> channels;
        split(hsv,channels);

        equalizeHist(channels[2], channels[2]);

        cv::Mat result;
        merge(channels,hsv);

        cvtColor(hsv,result,CV_HSV2BGR);
        //cvtColor(hsv,result,CV_BGR2HSV);

        return result;
    }
    return cv::Mat();
 }

 //std_msgs::ColorRGBA computeROIStats(const cv::Mat& inputImage, const cv::Rect& roi, cv::Scalar& mean, cv::Scalar& sd) 
 rock_publisher::colorRGBA computeROIStats(const cv::Mat& inputImage, const cv::Rect& roi, cv::Scalar& mean, cv::Scalar& sd, bool needHSV = false) 
 {
    cv::Mat roiImg = inputImage(roi);

    // do this if we need to convert to HSV
    if(needHSV)
    {
	cvtColor(roiImg, roiImg, CV_BGR2HSV,  CV_8U);
    }
	
    meanStdDev(roiImg, mean,sd);
    //std_msgs::ColorRGBA c;
    rock_publisher::colorRGBA c;
    c.r = mean[0]; c.g = mean[1]; c.b = mean[2]; c.a = 1.0;

    return c;  
 }

 void recalibrateCallback(const rock_publisher::recalibrateMsg& msg)
 {
     ROS_INFO("got recalibration msg"); 
     cv::Mat currentFrame;
     cv::Rect box;
     cv::Scalar mean;
     cv::Scalar sd;
     rock_publisher::colorRGBA color;
     // recalibrate color index is stored in color field cuz we are clever
     // eg. for color index 1, r = g = b= 1
     int colorToRecalibrate = msg.data.color.r;

     if( colorToRecalibrate == -1)
     {
	// reset total calibration
     	ROS_INFO("Reverting to default calibration. bye!"); 
	printCalibrationValues();
        restoreDefaultCalib();
	printCalibrationValues();
     } 
     else if( colorToRecalibrate < numColors_)
     {
	// need to convert ROI image to HSV before computing status.
	bool needHSV = true;
    	// do this 

     	try {
       	   currentFrame = cv::imdecode(cv::Mat(msg.img.data),1);
        }
        catch(cv::Exception& e)
        {
           ROS_ERROR("Couldn't decode compressed calibration image.\n %s", e.what());
	   return;
        }

	// TEST CASE: simple image 
	ROS_INFO("Bbox dimensions (x/y/w/h): %d %d %d %d", msg.data.x, msg.data.y, msg.data.width, msg.data.height);
     	box = cv::Rect(msg.data.x, msg.data.y, msg.data.width, msg.data.height);
/*	
	std::vector<cv::Mat> rgb;
	cv::Mat rg =  cv::Mat::zeros(200, 200, CV_8U);
	cv::Mat bl =  255 * cv::Mat::ones(200, 200, CV_8U);
	rgb.push_back(bl);		
	rgb.push_back(rg);	
	//rgb.push_back(rg);	
	rgb.push_back(bl/2);	
	merge(rgb, currentFrame);
     	cv::imwrite("/home/csrobot/blue.jpg", currentFrame);
*/		
     	//cv::imwrite("/home/csrobot/recal_img.jpg", currentFrame);
        //ROS_INFO("saved recal image");
//	cv::imshow("test image", currentFrame);
 //       cv::waitKey(1);	

	// get HSV color status.     
        color = computeROIStats(currentFrame, box, mean, sd, needHSV);

//	sd = cv::Scalar(10,10,10);
        
        sd[0] *= 2; sd[1] *= 2; sd[2] *= 2;
	cv::Scalar hsvMin = mean - sd; // add param for this.	
	cv::Scalar hsvMax = mean + sd; 	
	
	ROS_INFO("Changed color %d HSV thresholds:", colorToRecalibrate);
	ROS_INFO("H mean = %f H stdev = %f", mean[0], sd[0]);
	ROS_INFO("NEW: hsvMin = %f %f %f hsvMax = %f %f %f", hsvMin[0], hsvMin[1], hsvMin[2],  hsvMax[0], hsvMax[1], hsvMax[2]);
	
        // add calibration to thresholds vectors
	// before update 
	printCalibrationValues();

	mins[colorToRecalibrate].push_back(hsvMin);
	maxs[colorToRecalibrate].push_back(hsvMax);

	// after update 
	printCalibrationValues();

     }
     else
     {
     	ROS_ERROR("Got invalid recalibration color."); 
     }
 }

 void restoreDefaultCalib()
 {
      for(int j=0; j<mins.size(); ++j)
      {
         mins[j].erase(mins[j].begin()+1, mins[j].end());
         maxs[j].erase(maxs[j].begin()+1, maxs[j].end());
       }
 }

 void printCalibrationValues()
 {
       ROS_INFO("----HSV Calibration Threshs ---");
        
       for(int j=0; j<mins.size(); ++j)
       {
          std::vector<cv::Scalar> vec =  mins[j];
          cv::Scalar num = vec.back(); // top of calibration value stack
          ROS_INFO("index %d: min hsv : %f %f %f", j, num[0], num[1], num[2]);
          vec =  maxs[j];
          num = vec.back(); // top of calibration stack
          ROS_INFO("index %d: max hsv : %f %f %f", j, num[0], num[1], num[2]);
       }
}
}; 

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rover_cam_detect");
  RockDetection rd;
  ros::spin();
  //ros::spinOnce();
  return 0;
}
