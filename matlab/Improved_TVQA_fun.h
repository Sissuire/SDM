#ifndef _IMPROVED_TVQA_FUN_H_
#define _IMPROVED_TVQA_FUN_H_

#include "Improved_TVQA_def.h"

void TVQA(VideoInfo &videoInfo, std::vector<std::vector<double>> &quality);

void TVQA_optflo(VideoInfo &videoInfo, std::list<std::vector<double>> &ret_orig, std::list<std::vector<double>> &ret_dist, const int &RofCell, const int &RofRegion, const int &NofBins);

int TVQA_optflo_single(VideoInfo &videoInfo, std::list<std::vector<double>> &ret_orig, const int &RofCell, const int &RofRegion, const int &NofBins);
        
namespace version_1
{
	void BuildDescMat(const cv::Mat &xComp, const cv::Mat &yComp, float *desc, const DescInfo &descInfo);
	void HofComp(const cv::Mat &flow, float *desc, const DescInfo &descInfo);
	void GetDesc(const DescMat &descMat, const DescInfo &descInfo, std::vector<double> &desc);
}

namespace Farneback
{
	static void
		FarnebackUpdateMatrices(const cv::Mat& _R0, const cv::Mat& _R1, const cv::Mat& _flow, cv::Mat& matM, int _y0, int _y1);

	static void
		FarnebackUpdateFlow_Blur(const cv::Mat& _R0, const cv::Mat& _R1,
		cv::Mat& _flow, cv::Mat& matM, int block_size,
		bool update_matrices);

	static void
		FarnebackPrepareGaussian(int n, double sigma, float *g, float *xg, float *xxg,
		double &ig11, double &ig03, double &ig33, double &ig55);

	static void
		FarnebackPolyExp(const cv::Mat& src, cv::Mat& dst, int n, double sigma);

	void FarnebackCalc(cv::Mat &prev0, cv::Mat &next0, std::vector<cv::Mat> &flows, const FlowInfo &flowInfo);
};

void GetVideos(const std::string &fileName, std::vector<cv::Mat> &frames,
	const unsigned int &width, const unsigned int &height, const unsigned int &num, const unsigned int &fps);

void GetRect(const cv::Point &point, cv::Rect &rect, const int width, const int height, const DescInfo & descInfo);

void BuildDescMat(const cv::Mat &xComp, const cv::Mat &yComp, float *desc, const DescInfo &descInfo);

void GetDesc(const DescMat &descMat, const cv::Rect &rect, const DescInfo &descInfo, std::vector<float> &desc, const int index);

int ComputeDesc(const DescMat &descMat, const DescInfo &descInfo, std::vector<double> &desc);

void ReadCommonVideos(const std::string &fileName, std::vector<cv::Mat> &frames, const unsigned int &num);

void ReadYUV420p(const std::string &fileName, std::vector<cv::Mat> &frames,
	const unsigned int &width, const unsigned int &height, const unsigned int &num, const unsigned int &fps);

void GetOrigYUV420p(const std::string &fileName, std::vector<cv::Mat> &frames,
	const unsigned int &width, const unsigned int &height, const unsigned int &noof, const unsigned int &fps);

void ComputeSimpleGradient(const cv::Mat &img, cv::Mat &xComp, cv::Mat &yComp);

void HofComp(const cv::Mat &flow, float *desc, const DescInfo &descInfo);

void ComputeGradient(const cv::Mat &img, cv::Mat &gradient);

void ComputeGMSM(const cv::Mat &org, const cv::Mat &dst, cv::Mat &grad_chg);

void ComputeGMS(const cv::Mat &org, const cv::Mat &dst, std::vector<cv::Mat> &chg);

void ComputeGMSD(const cv::Mat &org, const cv::Mat &dst, std::vector<double> &quality, const int &num);

void ComputeGMSD(const std::vector<cv::Mat> &org, const std::vector<cv::Mat> &dst, std::vector<std::vector<double>> &quality, const int &num);

void ComputeGMSD_3D(const std::vector<cv::Mat> &org, const std::vector<cv::Mat> &dst, std::vector<double> &quality, const int &num);

void CenterBias(cv::Mat &mask, const double &maxVal, const double &minVal);

void Compute3DGM_Sobel(const std::vector<cv::Mat> &frames, cv::Mat &gm, cv::Mat &xComp, cv::Mat &yComp, cv::Mat &tComp, const size_t &num);

void Compute3DGM(const std::vector<cv::Mat> &frames, cv::Mat &gm, cv::Mat &xComp, cv::Mat &yComp, cv::Mat &tComp, const size_t &num);

void DenseSample(const cv::Mat& frame, const cv::Mat &mask, std::list<Track> &track, const TrackInfo &trackInfo, const DescInfo &hofInfo, const int &sample_step, const int &frame_num);

void ForegroundSampling(const cv::Mat& frame, const cv::Mat &saliencyMask, 
	std::list<Track> &track, const TrackInfo &trackInfo, const DescInfo &hofInfo, const int &sample_step, const int &frame_num);

bool IsValid(const std::vector<cv::Point> &trajectory, const std::vector<cv::Point2f> &displacement);

bool IsOverlapping(const std::vector<cv::Point> &traj, std::list<std::vector<cv::Point>> &region_inhibit, const int &len);

void CalcFlowWithFarneback(const cv::Mat &prev, const cv::Mat &curr, cv::Mat &flow0, const FlowInfo &flowInfo);

void ComputeDissimilarity(const std::vector<float> &vec1, const std::vector<float> &vec2, std::list<double> &tQuality);

void ComputeDissimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2, std::list<double> &tQuality);

double GetMeanVal(const std::list<double> &lst);

double GetMeanVal(const std::vector<double> &vec);

double SimpleTemporalPooling(const std::list<double> &tq);

double SimpleTemporalPooling(const std::vector<double> &tq);

void GetCube(const std::vector<cv::Mat> &frames, std::vector<cv::Mat> &cube, const std::vector<cv::Rect> &rects, const int &start_, const int &end_);

void GetScore_t(const std::vector<float>& desc_org, const std::vector<float>& desc_dst,
	const DescInfo &descInfo, const TrackInfo &trackInfo, std::list<double> &tQuality);

void GetScore_s(std::list<double> &sQuality, const std::vector<double> &sFrame, const int &start, const int &end);

void GetScore_st(const std::vector<cv::Mat> &OrigCube, const std::vector<cv::Mat> &DistCube, std::list<double> &tST);

void run(double *ptr_prev, double *ptr_next, const int rows, const int cols, std::vector<double> &desc_out);

void detect_kp(double *ptr, cv::Mat &out, const int rows, const int cols);
#endif