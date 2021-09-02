#include "Improved_TVQA_def.h"

int		traj_length = 18;		// trajectories length

/*
Farnback Optical Flow Parameters Setting
----------------------------------------
1. set " fb_pyr_scale = 0.5 " will get a faster computation

2. set " fb_levels = 4; fb_iterations = 5" will get a better performance, but a slower computation.

3. a good trade-off is as follows
*/

double	fb_pyr_scale = 1. / sqrt(2.);
int		fb_levels = 3;
int		fb_winsize = 10;
int		fb_iterations = 2;
int		fb_poly_n = 7;
double	fb_pol_sigma = 1.5;
int		fb_flags = 0;

/* dense points setting */
double	sample_quality = 0.05;		// quality for threshold, thre = maxmum * quality.
double	init_gap_r = 0.5;		// re-dense sampling
int		sample_rate = 5;		// dense sample step.
int		init_size = 5;		// minimum size of points for re-dense-sample.
int		eig_size = 3;

/* descriptors setting */
int		patch_size = 48;
int		nxy_cell = 2;
float	epsilon = 0.05f;
float	min_flow = 0.4f;
int		histNBins = 8;

/* check trajectory */
float	min_var = sqrt(3.f);
float	max_var = 10.f;
float	max_dis = 10.f;
float	static_thre = 1.f;
float	inhibit_quality = 0.4f;


float arg_c = 255.f;				// GMSD constant
double arg_T = 0.00001f;			// Dissimilarity constant


TrackInfo::TrackInfo(const int &len_, const int &gap_, const int &thre_)
	:length(len_), gap(gap_), thre_size(thre_){}

TrackInfo::~TrackInfo(){}

DescInfo::DescInfo(const int &patch_width, const int &patch_height, const int &nx, const int &ny, const int &bins)
	: width(patch_width), height(patch_height), nxCells(nx), nyCells(ny), nBins(bins), dims(bins * nx * ny){}

DescInfo::~DescInfo(){}

DescMat::DescMat(const int &width_, const int &height_, const int &nBins_)
	: width(width_), height(height_), nBins(nBins_), dims(width_ * height_ * nBins_)
{
	desc = (float*)malloc(dims * sizeof(float));
	memset(this->desc, 0, dims * sizeof(float));
}

DescMat::~DescMat(){}

void DescMat::Release(){ free(this->desc); }

Track::Track(const cv::Point &point_, const TrackInfo &trackInfo, const DescInfo &hofInfo, const int &frame_num)
	:point(trackInfo.length + 1, cv::Point(0, 0)), disp(trackInfo.length, cv::Point2f((float)0., (float)0.)),
	OrigHof(hofInfo.dims, (float)0.), DistHof(hofInfo.dims, (float)0.)
{
	rects.reserve(trackInfo.length + 1);
	end_frame = start_frame = frame_num;
	point[index] = point_;
}

void Track::addPoint(const cv::Point &point_)
{
	++end_frame;
	point[++index] = point_;
}
Track::~Track(){}

VideoInfo::VideoInfo(const std::string file_Orig_, const std::string file_Dist_, int width_, const int height_, const int frameNo_)
	:file_Orig(file_Orig_), file_Dist(file_Dist_), width(width_), height(height_), frameNo(frameNo_) {}

VideoInfo::~VideoInfo(){}

FlowInfo::FlowInfo(const double &scale_, const int &levels_, const int &winSize_, const int &iteration_, const int &poly_n_, const double &poly_sigma_)
	: scale(scale_), levels(levels_), winSize(winSize_), iteration(iteration_), poly_n(poly_n_), poly_sigma(poly_sigma_){}

FlowInfo::~FlowInfo(){}

