#include "Improved_TVQA_def.h"
#include "Improved_TVQA_fun.h"

bool writeOpticalFlow(const std::string& path, cv::InputArray flow)
{
	//    CV_Assert(sizeof(float) == 4);

	const float FLOW_TAG_FLOAT = 202021.25f;
	const char *FLOW_TAG_STRING = "PIEH";
	const int nChannels = 2;

	cv::Mat input = flow.getMat();
	if (input.channels() != nChannels || input.depth() != CV_32F || path.length() == 0)
		return false;

	std::ofstream file(path.c_str(), std::ofstream::binary);
	if (!file.good())
		return false;

	int nRows, nCols;

	nRows = (int)input.size().height;
	nCols = (int)input.size().width;

	const int headerSize = 12;
	char header[headerSize];
	memcpy(header, FLOW_TAG_STRING, 4);
	// size of ints is known - has been asserted in the current function
	memcpy(header + 4, reinterpret_cast<const char*>(&nCols), sizeof(nCols));
	memcpy(header + 8, reinterpret_cast<const char*>(&nRows), sizeof(nRows));
	file.write(header, headerSize);
	if (!file.good())
		return false;

	//    if ( input.isContinuous() ) //matrix is continous - treat it as a single row
	//    {
	//        nCols *= nRows;
	//        nRows = 1;
	//    }

	int row;
	char* p;
	for (row = 0; row < nRows; row++)
	{
		p = input.ptr<char>(row);
		file.write(p, nCols * nChannels * sizeof(float));
		if (!file.good())
			return false;
	}
	file.close();
	return true;
}


void TVQA(VideoInfo &videoInfo, std::vector<std::vector<double>> &quality)
{
	/*
	args:
	- videoInfo	: a data structure containing basic information about the videos.(width, height, file name, ...)
	- quality		: to save the final quality: q_ovearll, q_spatial, q_opticalFlow, q_spatio-temporal
	*/

	std::list<Track> tracker;

	int trajLength = traj_length;

	int initGap = int((float)trajLength * init_gap_r); // gap for initialize points
	initGap = int((float)trajLength);

	std::vector<cv::Mat> Orig, Dist;
	GetVideos(videoInfo.file_Orig, Orig, videoInfo.width, videoInfo.height, videoInfo.frameNo, 0);
	GetVideos(videoInfo.file_Dist, Dist, videoInfo.width, videoInfo.height, videoInfo.frameNo, 0);

	cv::Mat flowOrig, flowDist;

	/* initialize info.(about tracking, hof, and flow) */
	TrackInfo trackInfo(trajLength, initGap, init_size);
	DescInfo hofInfo(patch_size, patch_size, nxy_cell, nxy_cell, histNBins);
	FlowInfo flowInfo(fb_pyr_scale, fb_levels, fb_winsize, fb_iterations, fb_poly_n, fb_pol_sigma);

	cv::Mat mask(Orig[0].size(), CV_32FC1);
	CenterBias(mask, .9, 0.);

	/* sample points for the first frame, saving in tracker */
	//DenseSample(Dist[0], mask, tracker, trackInfo, hofInfo, sample_rate, 0);
	
	
	/* --------------------------------------------------------------------------------------- */
	std::string root_path_saliency = "J:/VQA_SourceCode/Consistent-video-saliency/Consistent-video-saliency/data/output-CuttingEdge/";
	std::string strSplit = videoInfo.file_Orig.substr(videoInfo.file_Orig.find_last_of('/') + 1);
	std::string file_name =  strSplit.substr(0, strSplit.find_last_of('.'));
	cv::Mat saliency_map0 = cv::imread(root_path_saliency + file_name + "/saliency/" + file_name + "_1.bmp", 0);

	cv::copyMakeBorder(saliency_map0, saliency_map0, 4, 4, 4, 4, cv::BORDER_REPLICATE);
	saliency_map0.convertTo(saliency_map0, CV_32FC1);
	cv::Mat saliency_map;
	if (!((mask.rows == saliency_map0.rows) && (mask.cols == saliency_map0.cols)))
	{
		saliency_map0(cv::Rect(0, 0, mask.cols, mask.rows)).copyTo(saliency_map);
	}
	else
		saliency_map0.copyTo(saliency_map);

	if (saliency_map.isContinuous())
	{
		cv::multiply(mask, saliency_map, saliency_map);
		cv::normalize(saliency_map, saliency_map, 1, 0, cv::NORM_MINMAX);
		ForegroundSampling(Dist[0], saliency_map, tracker, trackInfo, hofInfo, 2 * sample_rate, 0);
	}
	else
	{
		std::cout << "no Continuous!" << std::endl;
		return;
	}
	/* --------------------------------------------------------------------------------------- */

	std::list<double> sQualityInList, tQualityInList, stQualityInList;
	int dims = int(Orig.size());
	std::vector<double> spatials(dims, 0.);

	int start_ = 0, end_ = 0;
	int width = 0;
	int height = 0;

	bool flag_st = true;
	int init_counter = 0; // counter for re-sampling points

	for (int frame_num = 1; frame_num < dims - 1; ++frame_num)
	{
		ComputeGMSD(Orig[frame_num], Dist[frame_num], spatials, frame_num);

		/* compute optical flow */
		CalcFlowWithFarneback(Orig[frame_num], Orig[frame_num + 1], flowOrig, flowInfo);
		CalcFlowWithFarneback(Dist[frame_num], Dist[frame_num + 1], flowDist, flowInfo);

		width = flowOrig.cols;
		height = flowOrig.rows;
		++init_counter;

		/* Compute Integral Histograms for Optical Flow */
		DescMat hofOrig(width + 1, height + 1, hofInfo.nBins);
		DescMat hofDist(width + 1, height + 1, hofInfo.nBins);
		HofComp(flowOrig, hofOrig.desc, hofInfo);
		HofComp(flowDist, hofDist.desc, hofInfo);

		std::list<double> tHof, tST;

		bool valid_flag = false; // set it 'True' if tracking finished.
		int index = 0;
		int x, y;
		float dx, dy;
		cv::Rect rect;
		std::list<std::vector<cv::Point>> region_inhibit; // for removing two near-by trajectories.


		/* ---------------------------- */
		cv::Mat ttemp;
		Orig[frame_num].convertTo(ttemp, CV_8UC1);
		/* ---------------------------- */

		for (auto iTrack = tracker.begin(); iTrack != tracker.end();)
		{
			index = iTrack->index;
			cv::Point prevPoint = iTrack->point[index], currPoint;

			/* previous point */
			x = std::min<int>(std::max<int>(prevPoint.x, 0), width - 1);
			y = std::min<int>(std::max<int>(prevPoint.y, 0), height - 1);

			/* flow offset */
			dx = flowDist.ptr<float>(y)[2 * x];
			dy = flowDist.ptr<float>(y)[2 * x + 1];

			/* predicting current point */
			currPoint.x = prevPoint.x + cvRound(dx);
			currPoint.y = prevPoint.y + cvRound(dy);

			/* remove point out of boundary, or with a large displacement */
			if (currPoint.x <= 0 || currPoint.y <= 0 || currPoint.x >= width || currPoint.y >= height
				|| dx > max_dis || dx < -max_dis || dy > max_dis || dy < -max_dis)
			{
				iTrack = tracker.erase(iTrack);
				continue;
			}

			/* add displacement if current point is valid */
			iTrack->disp[index].x = dx;
			iTrack->disp[index].y = dy;

			/* add Hof features */
			GetRect(prevPoint, rect, width, height, hofInfo);
			GetDesc(hofOrig, rect, hofInfo, iTrack->OrigHof, index);
			GetDesc(hofDist, rect, hofInfo, iTrack->DistHof, index);

			/* push rect */
			iTrack->rects.push_back(rect);

			/* add current point */
			iTrack->addPoint(currPoint);

			/* check if tracking finished */
			if (iTrack->index == trackInfo.length)
			{
				if (IsValid(iTrack->point, iTrack->disp)) // False, if displacement of trajectories are random, or the trajectories are static.
				{
					if (IsOverlapping(iTrack->point, region_inhibit, trackInfo.length)) // False, if the trajectory is close to some one already existing.
					{
						valid_flag = true;
						start_ = iTrack->start_frame;
						end_ = iTrack->end_frame;

						std::vector<cv::Mat> OrigRect, DistRect;
						OrigRect.reserve(trackInfo.length);
						DistRect.reserve(trackInfo.length);
						GetCube(Orig, OrigRect, iTrack->rects, start_, end_);
						GetCube(Dist, DistRect, iTrack->rects, start_, end_);
						GetScore_st(OrigRect, DistRect, tST);

						GetScore_t(iTrack->OrigHof, iTrack->DistHof, hofInfo, trackInfo, tHof); // compute dissimilarity between two Hof features.

						/* ---------------------- */
						//cv::rectangle(ttemp, cv::Rect(iTrack->point[iTrack->index].x, iTrack->point[iTrack->index].y, 48, 48), cv::Scalar(0, 0, 255), 2);
						/* ---------------------- */
					}
				}
				iTrack = tracker.erase(iTrack); // if calculate the dissimilarity of the trajectory, then, remove it.
				continue;
			}
			++iTrack;
		}


		if (valid_flag) // if there's temporal quality computation, i.e., some trajectories finished, we convert these dissimilarity into spatial/temporal/combing quality. 
		{
			tQualityInList.push_back(SimpleTemporalPooling(tHof)); // temporal pooling 
			GetScore_s(sQualityInList, spatials, start_ + 1, end_ + 1); // get the spatial quality 
			stQualityInList.push_back(SimpleTemporalPooling(tST));

			/* --------------------------------- */
			//cv::imshow("frame", ttemp);
			//cv::waitKey(500);
			/* --------------------------------- */
		}

		region_inhibit.clear();
		hofOrig.Release();
		hofDist.Release();
		tHof.clear();
		tST.clear();

		if ((init_counter == trackInfo.gap) || (tracker.size() < trackInfo.thre_size)) // re-sample points
			//if(valid_flag)
		{
			//DenseSample(Dist[frame_num], mask, tracker, trackInfo, hofInfo, sample_rate, frame_num);
			
			/* --------------------------------------------------------------------------------------- */
			int num_here = frame_num / 3 + 1;
			saliency_map0 = cv::imread(root_path_saliency + file_name + "/saliency/" + file_name + "_" + std::to_string(num_here) + ".bmp", 0);

			cv::copyMakeBorder(saliency_map0, saliency_map0, 4, 4, 4, 4, cv::BORDER_REPLICATE);
			saliency_map0.convertTo(saliency_map0, CV_32FC1);

			if (!((mask.rows == saliency_map0.rows) && (mask.cols == saliency_map0.cols)))
			{
				saliency_map0(cv::Rect(0, 0, mask.cols, mask.rows)).copyTo(saliency_map);
			}
			else
				saliency_map0.copyTo(saliency_map);

			if (saliency_map.isContinuous())
			{
				cv::multiply(mask, saliency_map, saliency_map);
				cv::normalize(saliency_map, saliency_map, 1, 0, cv::NORM_MINMAX);
				ForegroundSampling(Dist[0], saliency_map, tracker, trackInfo, hofInfo, 2 * sample_rate, frame_num);
			}
			else
			{
				std::cout << "no Continuous!" << std::endl;
				return;
			}
			/* --------------------------------------------------------------------------------------- */

			init_counter = 0;
		}
	}

	/* combine spatial, temporal as well as spatio-temporal qualities */
	/*
	double sizes = 0.;
	double sq, tq, stq, score;
	sq = tq = stq = score = 0.;

	for (auto iter_s = sQualityInList.begin(), iter_t = tQualityInList.begin(), iter_st = stQualityInList.begin();
		iter_s != sQualityInList.end(); ++iter_s, ++iter_t, ++iter_st)
	{
		sq += (*iter_s);
		tq += (*iter_t);
		stq += (*iter_st);
		score += (10000 * (*iter_s) * (*iter_t) * (*iter_st)); // since the product result is too small, let 10000 times it.
		sizes += 1.;
	}
	sq /= sizes;
	tq /= sizes;
	stq /= sizes;
	score /= sizes;

	quality.resize(4);
	quality[0] = score;
	quality[1] = sq;
	quality[2] = tq;
	quality[3] = stq;
	*/

	quality.resize(3);
	quality[0].resize(sQualityInList.size());
	quality[1].resize(sQualityInList.size());
	quality[2].resize(sQualityInList.size());

	size_t counter_ret = 0;
	for (auto iter_s = sQualityInList.begin(), iter_t = tQualityInList.begin(), iter_st = stQualityInList.begin();
		iter_s != sQualityInList.end(); ++iter_s, ++iter_t, ++iter_st, ++counter_ret)
	{
		quality[0][counter_ret] = (*iter_s);
		quality[1][counter_ret] = (*iter_t);
		quality[2][counter_ret] = (*iter_st);
	}

	return;
}

void TVQA_optflo(VideoInfo &videoInfo, std::list<std::vector<double>> &ret_orig, std::list<std::vector<double>> &ret_dist, const int &RofCell, const int &RofRegion, const int &NofBins)
{
	/*
	args:
	- videoInfo	: a data structure containing basic information about the videos.(width, height, file name, ...)
	- quality		: to save the final quality: q_ovearll, q_spatial, q_opticalFlow, q_spatio-temporal
	*/

	//std::list<Track> tracker;

	//int trajLength = traj_length;

	//int initGap = int((float)trajLength * init_gap_r); // gap for initialize points
	//initGap = int((float)trajLength);

	std::vector<cv::Mat> Orig, Dist;
	GetVideos(videoInfo.file_Orig, Orig, videoInfo.width, videoInfo.height, videoInfo.frameNo, 0);
	GetVideos(videoInfo.file_Dist, Dist, videoInfo.width, videoInfo.height, videoInfo.frameNo, 0);

	cv::Mat flowOrig, flowDist;

	/* initialize info.(about tracking, hof, and flow) */
	//TrackInfo trackInfo(trajLength, initGap, init_size);
	DescInfo hofInfo(RofRegion, RofRegion, RofCell, RofCell, NofBins);
	FlowInfo flowInfo(fb_pyr_scale, fb_levels, fb_winsize, fb_iterations, fb_poly_n, fb_pol_sigma);

	//cv::Mat mask(Orig[0].size(), CV_32FC1);
	//CenterBias(mask, .9, 0.);

	/* sample points for the first frame, saving in tracker */
	//DenseSample(Dist[0], mask, tracker, trackInfo, hofInfo, sample_rate, 0);


	/* --------------------------------------------------------------------------------------- */
	/*std::string root_path_saliency = "J:/VQA_SourceCode/Consistent-video-saliency/Consistent-video-saliency/data/output-CuttingEdge/";
	std::string strSplit = videoInfo.file_Orig.substr(videoInfo.file_Orig.find_last_of('/') + 1);
	std::string file_name = strSplit.substr(0, strSplit.find_last_of('.'));
	cv::Mat saliency_map0 = cv::imread(root_path_saliency + file_name + "/saliency/" + file_name + "_1.bmp", 0);

	cv::copyMakeBorder(saliency_map0, saliency_map0, 4, 4, 4, 4, cv::BORDER_REPLICATE);
	saliency_map0.convertTo(saliency_map0, CV_32FC1);
	cv::Mat saliency_map;
	if (!((mask.rows == saliency_map0.rows) && (mask.cols == saliency_map0.cols)))
	{
		saliency_map0(cv::Rect(0, 0, mask.cols, mask.rows)).copyTo(saliency_map);
	}
	else
		saliency_map0.copyTo(saliency_map);

	if (saliency_map.isContinuous())
	{
		cv::multiply(mask, saliency_map, saliency_map);
		cv::normalize(saliency_map, saliency_map, 1, 0, cv::NORM_MINMAX);
		ForegroundSampling(Dist[0], saliency_map, tracker, trackInfo, hofInfo, 2 * sample_rate, 0);
	}
	else
	{
		std::cout << "no Continuous!" << std::endl;
		return;
	}*/
	/* --------------------------------------------------------------------------------------- */

	//std::list<double> sQualityInList, tQualityInList, stQualityInList;
	int dims = int(Orig.size());
	//std::vector<double> spatials(dims, 0.);

	//int start_ = 0, end_ = 0;
	int width = 0;
	int height = 0;

	//bool flag_st = true;
	//int init_counter = 0; // counter for re-sampling points

	std::vector<double> desc;
	for (int frame_num = 0; frame_num < dims - 1; ++frame_num)
	{
		//ComputeGMSD(Orig[frame_num], Dist[frame_num], spatials, frame_num);
		//ComputeGMSD_3D(Orig, Dist, spatials, frame_num);

		/* compute optical flow */
		CalcFlowWithFarneback(Orig[frame_num], Orig[frame_num + 1], flowOrig, flowInfo);
		CalcFlowWithFarneback(Dist[frame_num], Dist[frame_num + 1], flowDist, flowInfo);

		width = flowOrig.cols;
		height = flowOrig.rows;
		//++init_counter;

		/* Compute Integral Histograms for Optical Flow */
		DescMat hofOrig(width + 1, height + 1, hofInfo.nBins);
		DescMat hofDist(width + 1, height + 1, hofInfo.nBins);
		HofComp(flowOrig, hofOrig.desc, hofInfo);
		HofComp(flowDist, hofDist.desc, hofInfo);

		/* ------------------------------------------- */
		desc.clear();
		ComputeDesc(hofOrig, hofInfo, desc);
		ret_orig.push_back(desc);
        hofOrig.Release();

		desc.clear();
		ComputeDesc(hofDist, hofInfo, desc);
		ret_dist.push_back(desc);
        hofDist.Release();
		/* ------------------------------------------- */



		//std::list<double> tHof, tST;

		//bool valid_flag = false; // set it 'True' if tracking finished.
		//int index = 0;
		//int x, y;
		//float dx, dy;
		//cv::Rect rect;
		//std::list<std::vector<cv::Point>> region_inhibit; // for removing two near-by trajectories.


		/* ---------------------------- */
		//cv::Mat ttemp;
		//Orig[frame_num].convertTo(ttemp, CV_8UC1);
		/* ---------------------------- */
		//
		//for (auto iTrack = tracker.begin(); iTrack != tracker.end();)
		//{
		//	index = iTrack->index;
		//	cv::Point prevPoint = iTrack->point[index], currPoint;
		//
		//	/* previous point */
		//	x = std::min<int>(std::max<int>(prevPoint.x, 0), width - 1);
		//	y = std::min<int>(std::max<int>(prevPoint.y, 0), height - 1);
		//
		//	/* flow offset */
		//	dx = flowDist.ptr<float>(y)[2 * x];
		//	dy = flowDist.ptr<float>(y)[2 * x + 1];
		//
		//	/* predicting current point */
		//	currPoint.x = prevPoint.x + cvRound(dx);
		//	currPoint.y = prevPoint.y + cvRound(dy);
		//
		//	/* remove point out of boundary, or with a large displacement */
		//	if (currPoint.x <= 0 || currPoint.y <= 0 || currPoint.x >= width || currPoint.y >= height
		//		|| dx > max_dis || dx < -max_dis || dy > max_dis || dy < -max_dis)
		//	{
		//		iTrack = tracker.erase(iTrack);
		//		continue;
		//	}
		//
		//	/* add displacement if current point is valid */
		//	iTrack->disp[index].x = dx;
		//	iTrack->disp[index].y = dy;
		//
		//	/* add Hof features */
		//	GetRect(prevPoint, rect, width, height, hofInfo);
		//	GetDesc(hofOrig, rect, hofInfo, iTrack->OrigHof, index);
		//	GetDesc(hofDist, rect, hofInfo, iTrack->DistHof, index);
		//
		//	/* push rect */
		//	iTrack->rects.push_back(rect);
		//
		//	/* add current point */
		//	iTrack->addPoint(currPoint);
		//
		//	/* check if tracking finished */
		//	if (iTrack->index == trackInfo.length)
		//	{
		//		if (IsValid(iTrack->point, iTrack->disp)) // False, if displacement of trajectories are random, or the trajectories are static.
		//		{
		//			if (IsOverlapping(iTrack->point, region_inhibit, trackInfo.length)) // False, if the trajectory is close to some one already existing.
		//			{
		//				valid_flag = true;
		//				start_ = iTrack->start_frame;
		//				end_ = iTrack->end_frame;
		//
		//				/*std::vector<cv::Mat> OrigRect, DistRect;
		//				OrigRect.reserve(trackInfo.length);
		//				DistRect.reserve(trackInfo.length);
		//				GetCube(Orig, OrigRect, iTrack->rects, start_, end_);
		//				GetCube(Dist, DistRect, iTrack->rects, start_, end_);
		//				GetScore_st(OrigRect, DistRect, tST);*/
		//
		//				GetScore_t(iTrack->OrigHof, iTrack->DistHof, hofInfo, trackInfo, tHof); // compute dissimilarity between two Hof features.
		//
		//				/* ---------------------- */
		//				//cv::rectangle(ttemp, cv::Rect(iTrack->point[iTrack->index].x, iTrack->point[iTrack->index].y, 48, 48), cv::Scalar(0, 0, 255), 2);
		//				/* ---------------------- */
		//			}
		//		}
		//		iTrack = tracker.erase(iTrack); // if calculate the dissimilarity of the trajectory, then, remove it.
		//		continue;
		//	}
		//	++iTrack;
		//}


		//if (valid_flag) // if there's temporal quality computation, i.e., some trajectories finished, we convert these dissimilarity into spatial/temporal/combing quality. 
		//{
		//	tQualityInList.push_back(SimpleTemporalPooling(tHof)); // temporal pooling 
		//	GetScore_s(sQualityInList, spatials, start_ + 1, end_ + 1); // get the spatial quality 
		//	stQualityInList.push_back(SimpleTemporalPooling(tST));

			/* --------------------------------- */
			//cv::imshow("frame", ttemp);
			//cv::waitKey(500);
			/* --------------------------------- */
		//}

		//region_inhibit.clear();
		//hofOrig.Release();
		//hofDist.Release();
		//tHof.clear();
		//tST.clear();

		//if ((init_counter == trackInfo.gap) || (tracker.size() < trackInfo.thre_size)) // re-sample points
		//	//if(valid_flag)
		//{
		//	DenseSample(Dist[frame_num], mask, tracker, trackInfo, hofInfo, sample_rate, frame_num);
		//
		//	/* --------------------------------------------------------------------------------------- */
		//	/*int num_here = frame_num / 3 + 1;
		//	saliency_map0 = cv::imread(root_path_saliency + file_name + "/saliency/" + file_name + "_" + std::to_string(num_here) + ".bmp", 0);
		//
		//	cv::copyMakeBorder(saliency_map0, saliency_map0, 4, 4, 4, 4, cv::BORDER_REPLICATE);
		//	saliency_map0.convertTo(saliency_map0, CV_32FC1);
		//
		//	if (!((mask.rows == saliency_map0.rows) && (mask.cols == saliency_map0.cols)))
		//	{
		//		saliency_map0(cv::Rect(0, 0, mask.cols, mask.rows)).copyTo(saliency_map);
		//	}
		//	else
		//		saliency_map0.copyTo(saliency_map);
		//
		//	if (saliency_map.isContinuous())
		//	{
		//		cv::multiply(mask, saliency_map, saliency_map);
		//		cv::normalize(saliency_map, saliency_map, 1, 0, cv::NORM_MINMAX);
		//		ForegroundSampling(Dist[0], saliency_map, tracker, trackInfo, hofInfo, 2 * sample_rate, frame_num);
		//	}
		//	else
		//	{
		//		std::cout << "no Continuous!" << std::endl;
		//		return;
		//	}*/
		//	/* --------------------------------------------------------------------------------------- */
		//
		//	init_counter = 0;
		//}
	}

	/* combine spatial, temporal as well as spatio-temporal qualities */
	/*
	double sizes = 0.;
	double sq, tq, stq, score;
	sq = tq = stq = score = 0.;

	for (auto iter_s = sQualityInList.begin(), iter_t = tQualityInList.begin(), iter_st = stQualityInList.begin();
	iter_s != sQualityInList.end(); ++iter_s, ++iter_t, ++iter_st)
	{
	sq += (*iter_s);
	tq += (*iter_t);
	stq += (*iter_st);
	score += (10000 * (*iter_s) * (*iter_t) * (*iter_st)); // since the product result is too small, let 10000 times it.
	sizes += 1.;
	}
	sq /= sizes;
	tq /= sizes;
	stq /= sizes;
	score /= sizes;

	quality.resize(4);
	quality[0] = score;
	quality[1] = sq;
	quality[2] = tq;
	quality[3] = stq;
	*/

	//quality.resize(3);
	//quality[0].resize(sQualityInList.size());
	//quality[1].resize(sQualityInList.size());
	//quality[2].resize(sQualityInList.size());

	//size_t counter_ret = 0;
	//for (auto iter_s = sQualityInList.begin(), iter_t = tQualityInList.begin(), iter_st = stQualityInList.begin();
	//	iter_s != sQualityInList.end(); ++iter_s, ++iter_t, ++iter_st, ++counter_ret)
	//{
	//	quality[0][counter_ret] = (*iter_s);
	//	quality[1][counter_ret] = (*iter_t);
	//	quality[2][counter_ret] = (*iter_st);
	//}

	return;
}

int TVQA_optflo_single(VideoInfo &videoInfo, std::list<std::vector<double>> &ret_orig, const int &RofCell, const int &RofRegion, const int &NofBins)
{
	std::vector<cv::Mat> Orig;
	GetVideos(videoInfo.file_Orig, Orig, videoInfo.width, videoInfo.height, videoInfo.frameNo, 0);

	cv::Mat flowOrig;

	DescInfo hofInfo(RofRegion, RofRegion, RofCell, RofCell, NofBins);
	FlowInfo flowInfo(fb_pyr_scale, fb_levels, fb_winsize, fb_iterations, fb_poly_n, fb_pol_sigma);

	int dims = int(Orig.size());

	int width = 0;
	int height = 0;

    int blks;
    
	std::vector<double> desc;
	for (int frame_num = 0; frame_num < dims - 1; ++frame_num)
	{
		/* compute optical flow */
		CalcFlowWithFarneback(Orig[frame_num], Orig[frame_num + 1], flowOrig, flowInfo);

		width = flowOrig.cols;
		height = flowOrig.rows;

		/* Compute Integral Histograms for Optical Flow */
		DescMat hofOrig(width + 1, height + 1, hofInfo.nBins);
		HofComp(flowOrig, hofOrig.desc, hofInfo);

		desc.clear();
		blks = ComputeDesc(hofOrig, hofInfo, desc);
		ret_orig.push_back(desc);
        hofOrig.Release();
	}
    
    return blks;
}