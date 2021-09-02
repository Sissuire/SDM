#ifndef _IMPROVED_TVQA_DEF_H_
#define _IMPROVED_TVQA_DEF_H_

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <time.h>
#include <iomanip>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
//#include <opencv2/optflow.hpp>

extern int		traj_length;		// trajectories length

/* Farnback Optical Flow Parameters Setting */
extern double	fb_pyr_scale;
extern int		fb_levels;
extern int		fb_winsize;
extern int		fb_iterations;
extern int		fb_poly_n;
extern double	fb_pol_sigma;
extern int		fb_flags;

/* dense points setting */
extern double	sample_quality;		// quality for threshold, thre = maxmum * quality.
extern double	init_gap_r;		// re-dense sampling
extern int		sample_rate;		// dense sample step.
extern int		init_size;		// minimum size of points for re-dense-sample.
extern int		eig_size;

/* descriptors setting */
extern int		patch_size;
extern int		nxy_cell;
extern float	epsilon;
extern float	min_flow;
extern int		histNBins;

/* check trajectory */
extern float	min_var;
extern float	max_var;
extern float	max_dis;
extern float	static_thre;
extern float	inhibit_quality;


extern float arg_c;				// GMSD constant
extern double arg_T;			// Dissimilarity constant


class TrackInfo
{
public:
	const int length, gap;
	int thre_size;

	TrackInfo(const int &len_, const int &gap_, const int &thre_);

	~TrackInfo();
};

class DescInfo{
public:
	const int height, width;
	const int nxCells, nyCells;
	const int nBins;
	const int dims;

	DescInfo(const int &patch_width, const int &patch_height, const int &nx, const int &ny, const int &bins);

	~DescInfo();

};

class DescMat{
public:
	const int width, height;
	const int nBins;
	const int dims;

	float *desc;

	DescMat(const int &width_, const int &height_, const int &nBins_);

	~DescMat();

	void Release();
};

class Track
{
public:
	std::vector<cv::Point> point;
	std::vector<cv::Point2f> disp;

	std::vector<float> OrigHof, DistHof;
	std::vector<cv::Rect> rects;
	int index = 0;
	int start_frame = 0, end_frame = 0;

	Track(const cv::Point &point_, const TrackInfo &trackInfo, const DescInfo &hofInfo, const int &frame_num);
	~Track();
	void addPoint(const cv::Point &point_);
};

class VideoInfo
{
public:
	const std::string file_Orig;
	const std::string file_Dist;
	const int width, height;
	int frameNo;

	VideoInfo(const std::string file_Orig_, const std::string file_Dist_, int width_, const int height_, const int frameNo_);

	~VideoInfo();
};

class FlowInfo
{
public:
	const int levels;
	const int winSize;
	const int iteration;
	const int poly_n;

	const double poly_sigma;
	const double scale;

	FlowInfo(const double &scale_, const int &levels_, const int &winSize_, const int &iteration_, const int &poly_n_, const double &poly_sigma_);

	~FlowInfo();
};


#endif