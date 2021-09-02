#include "Improved_TVQA_fun.h"

namespace Farneback
{
	static void
		FarnebackUpdateMatrices(const cv::Mat& _R0, const cv::Mat& _R1, const cv::Mat& _flow, cv::Mat& matM, int _y0, int _y1)
	{
		const int BORDER = 5;
		static const float border[BORDER] = { 0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f };

		int x, y, width = _flow.cols, height = _flow.rows;
		const float* R1 = _R1.ptr<float>();
		size_t step1 = _R1.step / sizeof(R1[0]);

		matM.create(height, width, CV_32FC(5));

		for (y = _y0; y < _y1; y++)
		{
			const float* flow = _flow.ptr<float>(y);
			const float* R0 = _R0.ptr<float>(y);
			float* M = matM.ptr<float>(y);

			for (x = 0; x < width; x++)
			{
				float dx = flow[x * 2], dy = flow[x * 2 + 1];
				float fx = x + dx, fy = y + dy;

#if 1
				int x1 = cvFloor(fx), y1 = cvFloor(fy);
				const float* ptr = R1 + y1*step1 + x1 * 5;
				float r2, r3, r4, r5, r6;

				fx -= x1; fy -= y1;

				if ((unsigned)x1 < (unsigned)(width - 1) &&
					(unsigned)y1 < (unsigned)(height - 1))
				{
					float a00 = (1.f - fx)*(1.f - fy), a01 = fx*(1.f - fy),
						a10 = (1.f - fx)*fy, a11 = fx*fy;

					r2 = a00*ptr[0] + a01*ptr[5] + a10*ptr[step1] + a11*ptr[step1 + 5];
					r3 = a00*ptr[1] + a01*ptr[6] + a10*ptr[step1 + 1] + a11*ptr[step1 + 6];
					r4 = a00*ptr[2] + a01*ptr[7] + a10*ptr[step1 + 2] + a11*ptr[step1 + 7];
					r5 = a00*ptr[3] + a01*ptr[8] + a10*ptr[step1 + 3] + a11*ptr[step1 + 8];
					r6 = a00*ptr[4] + a01*ptr[9] + a10*ptr[step1 + 4] + a11*ptr[step1 + 9];

					r4 = (R0[x * 5 + 2] + r4)*0.5f;
					r5 = (R0[x * 5 + 3] + r5)*0.5f;
					r6 = (R0[x * 5 + 4] + r6)*0.25f;
				}
#else
				int x1 = cvRound(fx), y1 = cvRound(fy);
				const float* ptr = R1 + y1*step1 + x1 * 5;
				float r2, r3, r4, r5, r6;

				if ((unsigned)x1 < (unsigned)width &&
					(unsigned)y1 < (unsigned)height)
				{
					r2 = ptr[0];
					r3 = ptr[1];
					r4 = (R0[x * 5 + 2] + ptr[2])*0.5f;
					r5 = (R0[x * 5 + 3] + ptr[3])*0.5f;
					r6 = (R0[x * 5 + 4] + ptr[4])*0.25f;
				}
#endif
				else
				{
					r2 = r3 = 0.f;
					r4 = R0[x * 5 + 2];
					r5 = R0[x * 5 + 3];
					r6 = R0[x * 5 + 4] * 0.5f;
				}

				r2 = (R0[x * 5] - r2)*0.5f;
				r3 = (R0[x * 5 + 1] - r3)*0.5f;

				r2 += r4*dy + r6*dx;
				r3 += r6*dy + r5*dx;

				if ((unsigned)(x - BORDER) >= (unsigned)(width - BORDER * 2) ||
					(unsigned)(y - BORDER) >= (unsigned)(height - BORDER * 2))
				{
					float scale = (x < BORDER ? border[x] : 1.f)*
						(x >= width - BORDER ? border[width - x - 1] : 1.f)*
						(y < BORDER ? border[y] : 1.f)*
						(y >= height - BORDER ? border[height - y - 1] : 1.f);

					r2 *= scale; r3 *= scale; r4 *= scale;
					r5 *= scale; r6 *= scale;
				}

				M[x * 5] = r4*r4 + r6*r6; // G(1,1)
				M[x * 5 + 1] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
				M[x * 5 + 2] = r5*r5 + r6*r6; // G(2,2)
				M[x * 5 + 3] = r4*r2 + r6*r3; // h(1)
				M[x * 5 + 4] = r6*r2 + r5*r3; // h(2)
			}
		}
	}

	static void
		FarnebackUpdateFlow_Blur(const cv::Mat& _R0, const cv::Mat& _R1,
		cv::Mat& _flow, cv::Mat& matM, int block_size,
		bool update_matrices)
	{
		int x, y, width = _flow.cols, height = _flow.rows;
		int m = block_size / 2;
		int y0 = 0, y1;
		int min_update_stripe = std::max((1 << 10) / width, block_size);
		double scale = 1. / (block_size*block_size);

		cv::AutoBuffer<double> _vsum((width + m * 2 + 2) * 5);
		double* vsum = _vsum + (m + 1) * 5;

		// init vsum
		const float* srow0 = matM.ptr<float>();
		for (x = 0; x < width * 5; x++)
			vsum[x] = srow0[x] * (m + 2);

		for (y = 1; y < m; y++)
		{
			srow0 = matM.ptr<float>(std::min(y, height - 1));
			for (x = 0; x < width * 5; x++)
				vsum[x] += srow0[x];
		}

		// compute blur(G)*flow=blur(h)
		for (y = 0; y < height; y++)
		{
			double g11, g12, g22, h1, h2;
			float* flow = _flow.ptr<float>(y);

			srow0 = matM.ptr<float>(std::max(y - m - 1, 0));
			const float* srow1 = matM.ptr<float>(std::min(y + m, height - 1));

			// vertical blur
			for (x = 0; x < width * 5; x++)
				vsum[x] += srow1[x] - srow0[x];

			// update borders
			for (x = 0; x < (m + 1) * 5; x++)
			{
				vsum[-1 - x] = vsum[4 - x];
				vsum[width * 5 + x] = vsum[width * 5 + x - 5];
			}

			// init g** and h*
			g11 = vsum[0] * (m + 2);
			g12 = vsum[1] * (m + 2);
			g22 = vsum[2] * (m + 2);
			h1 = vsum[3] * (m + 2);
			h2 = vsum[4] * (m + 2);

			for (x = 1; x < m; x++)
			{
				g11 += vsum[x * 5];
				g12 += vsum[x * 5 + 1];
				g22 += vsum[x * 5 + 2];
				h1 += vsum[x * 5 + 3];
				h2 += vsum[x * 5 + 4];
			}

			// horizontal blur
			for (x = 0; x < width; x++)
			{
				g11 += vsum[(x + m) * 5] - vsum[(x - m) * 5 - 5];
				g12 += vsum[(x + m) * 5 + 1] - vsum[(x - m) * 5 - 4];
				g22 += vsum[(x + m) * 5 + 2] - vsum[(x - m) * 5 - 3];
				h1 += vsum[(x + m) * 5 + 3] - vsum[(x - m) * 5 - 2];
				h2 += vsum[(x + m) * 5 + 4] - vsum[(x - m) * 5 - 1];

				double g11_ = g11*scale;
				double g12_ = g12*scale;
				double g22_ = g22*scale;
				double h1_ = h1*scale;
				double h2_ = h2*scale;

				double idet = 1. / (g11_*g22_ - g12_*g12_ + 1e-3);

				flow[x * 2] = (float)((g11_*h2_ - g12_*h1_)*idet);
				flow[x * 2 + 1] = (float)((g22_*h1_ - g12_*h2_)*idet);
			}

			y1 = y == height - 1 ? height : y - block_size;
			if (update_matrices && (y1 == height || y1 >= y0 + min_update_stripe))
			{
				FarnebackUpdateMatrices(_R0, _R1, _flow, matM, y0, y1);
				y0 = y1;
			}
		}
	}

	static void
		FarnebackPrepareGaussian(int n, double sigma, float *g, float *xg, float *xxg,
		double &ig11, double &ig03, double &ig33, double &ig55)
	{
		if (sigma < FLT_EPSILON)
			sigma = n*0.3;

		double s = 0.;
		for (int x = -n; x <= n; x++)
		{
			g[x] = (float)std::exp(-x*x / (2 * sigma*sigma));
			s += g[x];
		}

		s = 1. / s;
		for (int x = -n; x <= n; x++)
		{
			g[x] = (float)(g[x] * s);
			xg[x] = (float)(x*g[x]);
			xxg[x] = (float)(x*x*g[x]);
		}

		cv::Mat_<double> G(6, 6);
		G.setTo(0);

		for (int y = -n; y <= n; y++)
		{
			for (int x = -n; x <= n; x++)
			{
				G(0, 0) += g[y] * g[x];
				G(1, 1) += g[y] * g[x] * x*x;
				G(3, 3) += g[y] * g[x] * x*x*x*x;
				G(5, 5) += g[y] * g[x] * x*x*y*y;
			}
		}

		//G[0][0] = 1.;
		G(2, 2) = G(0, 3) = G(0, 4) = G(3, 0) = G(4, 0) = G(1, 1);
		G(4, 4) = G(3, 3);
		G(3, 4) = G(4, 3) = G(5, 5);

		// invG:
		// [ x        e  e    ]
		// [    y             ]
		// [       y          ]
		// [ e        z       ]
		// [ e           z    ]
		// [                u ]
		cv::Mat_<double> invG = G.inv(cv::DECOMP_CHOLESKY);

		ig11 = invG(1, 1);
		ig03 = invG(0, 3);
		ig33 = invG(3, 3);
		ig55 = invG(5, 5);
	}

	static void
		FarnebackPolyExp(const cv::Mat& src, cv::Mat& dst, int n, double sigma)
	{
		int k, x, y;

		CV_Assert(src.type() == CV_32FC1);
		int width = src.cols;
		int height = src.rows;
		cv::AutoBuffer<float> kbuf(n * 6 + 3), _row((width + n * 2) * 3);
		float* g = kbuf + n;
		float* xg = g + n * 2 + 1;
		float* xxg = xg + n * 2 + 1;
		float *row = (float*)_row + n * 3;
		double ig11, ig03, ig33, ig55;

		FarnebackPrepareGaussian(n, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);

		dst.create(height, width, CV_32FC(5));

		for (y = 0; y < height; y++)
		{
			float g0 = g[0], g1, g2;
			const float *srow0 = src.ptr<float>(y), *srow1 = 0;
			float *drow = dst.ptr<float>(y);

			/* vertical part of convolution */
			for (x = 0; x < width; x++)
			{
				row[x * 3] = srow0[x] * g0;
				row[x * 3 + 1] = row[x * 3 + 2] = 0.f;
			}

			for (k = 1; k <= n; k++)
			{
				g0 = g[k]; g1 = xg[k]; g2 = xxg[k];
				srow0 = src.ptr<float>(std::max(y - k, 0));
				srow1 = src.ptr<float>(std::min(y + k, height - 1));

				for (x = 0; x < width; x++)
				{
					float p = srow0[x] + srow1[x];
					float t0 = row[x * 3] + g0*p;
					float t1 = row[x * 3 + 1] + g1*(srow1[x] - srow0[x]);
					float t2 = row[x * 3 + 2] + g2*p;

					row[x * 3] = t0;
					row[x * 3 + 1] = t1;
					row[x * 3 + 2] = t2;
				}
			}

			/* horizontal part of convolution */
			for (x = 0; x < n * 3; x++)
			{
				row[-1 - x] = row[2 - x];
				row[width * 3 + x] = row[width * 3 + x - 3];
			}

			for (x = 0; x < width; x++)
			{
				g0 = g[0];
				/* r1 ~ 1, r2 ~ x, r3 ~ y, r4 ~ x^2, r5 ~ y^2, r6 ~ xy */
				double b1 = row[x * 3] * g0, b2 = 0, b3 = row[x * 3 + 1] * g0,
					b4 = 0, b5 = row[x * 3 + 2] * g0, b6 = 0;

				for (k = 1; k <= n; k++)
				{
					double tg = row[(x + k) * 3] + row[(x - k) * 3];
					g0 = g[k];
					b1 += tg*g0;
					b4 += tg*xxg[k];
					b2 += (row[(x + k) * 3] - row[(x - k) * 3])*xg[k];
					b3 += (row[(x + k) * 3 + 1] + row[(x - k) * 3 + 1])*g0;
					b6 += (row[(x + k) * 3 + 1] - row[(x - k) * 3 + 1])*xg[k];
					b5 += (row[(x + k) * 3 + 2] + row[(x - k) * 3 + 2])*g0;
				}

				/* do not store r1 */
				drow[x * 5 + 1] = (float)(b2*ig11);
				drow[x * 5] = (float)(b3*ig11);
				drow[x * 5 + 3] = (float)(b1*ig03 + b4*ig33);
				drow[x * 5 + 2] = (float)(b1*ig03 + b5*ig33);
				drow[x * 5 + 4] = (float)(b6*ig55);
			}
		}

		row -= n * 3;
	}

	void FarnebackCalc(cv::Mat &prev0, cv::Mat &next0, std::vector<cv::Mat> &flows, const FlowInfo &flowInfo) // Calculate Farneback Optical Flow
	{
		const int min_size = 32;
		const cv::Mat* img[2] = { &prev0, &next0 };

		int i, k;
		double scale;
		cv::Mat prevFlow, flow, fimg;
		int levels = flowInfo.levels;

		for (k = 0, scale = 1; k < levels; k++)
		{
			scale *= flowInfo.scale; // pyrScale is smaller than 1., representing the scale factor of downsample
			if (prev0.cols*scale < min_size || prev0.rows*scale < min_size)
				break;
		}

		levels = k;

		for (k = levels; k >= 0; k--)
		{
			for (i = 0, scale = 1; i < k; i++)
				scale *= flowInfo.scale;

			double sigma = (1. / scale - 1)*0.5;
			int smooth_sz = cvRound(sigma * 5) | 1;
			smooth_sz = std::max(smooth_sz, 3);

			int width = cvRound(prev0.cols*scale);
			int height = cvRound(prev0.rows*scale);

			flow.create(height, width, CV_32FC2);


			if (prevFlow.empty())
			{
				flow = cv::Mat::zeros(height, width, CV_32FC2);
			}
			else
			{
				cv::resize(prevFlow, flow, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
				flow *= 1. / flowInfo.scale;
			}

			cv::Mat R[2], I, M;
			for (i = 0; i < 2; i++)
			{
				img[i]->convertTo(fimg, CV_32F);
				cv::GaussianBlur(fimg, fimg, cv::Size(smooth_sz, smooth_sz), sigma, sigma);
				cv::resize(fimg, I, cv::Size(width, height), cv::INTER_LINEAR);
				Farneback::FarnebackPolyExp(I, R[i], flowInfo.poly_n, flowInfo.poly_sigma);
			}

			Farneback::FarnebackUpdateMatrices(R[0], R[1], flow, M, 0, flow.rows);

			for (i = 0; i < flowInfo.iteration; i++)
			{
				Farneback::FarnebackUpdateFlow_Blur(R[0], R[1], flow, M, flowInfo.winSize, i < flowInfo.iteration - 1);
			}

			flow.copyTo(flows[k]);
			prevFlow = flow;
		}
	}
};

namespace version_1
{
	void BuildDescMat(const cv::Mat &xComp, const cv::Mat &yComp, float *desc, const DescInfo &descInfo)
	{
		/* compute integral histograms for the whole image */
		float maxAngle = 360.f;
		int nDims = descInfo.nBins;

		/* one more bin for hof */
		int nBins = descInfo.nBins - 1;
		const float angleBase = float(nBins) / maxAngle;

		int step = (xComp.cols + 1) * nDims;
		int index = step + nDims;
		for (int i = 0; i != xComp.rows; ++i, index += nDims)
		{
			const float* xc = xComp.ptr<float>(i);
			const float* yc = yComp.ptr<float>(i);

			/* summarization of the current line */
			std::vector<float> sum(nDims);
			for (int j = 0; j != xComp.cols; ++j)
			{
				float x = xc[j], y = yc[j];
				float mag0 = sqrt(x*x + y*y), mag1;
				int bin0, bin1;

				/* for the zero bin of hof */
				if (mag0 <= min_flow)
				{
					bin0 = nBins;	// the zero bin is the last one
					mag0 = 1.0;
					bin1 = 0;
					mag1 = 0;
				}
				else
				{
					float angle = cv::fastAtan2(y, x);
					if (angle >= maxAngle) angle -= maxAngle;

					/* split the mag to two adjacent bins */
					float fbin = angle * angleBase;
					bin0 = cvFloor(fbin);
					bin1 = (bin0 + 1) % nBins;

					mag1 = (fbin - bin0) * mag0;
					mag0 -= mag1;
				}

				sum[bin0] += mag0;
				sum[bin1] += mag1;

				for (int m = 0; m != nDims; ++m, ++index)
					desc[index] = desc[index - step] + sum[m];
			}
		}
	}

	void HofComp(const cv::Mat &flow, float *desc, const DescInfo &descInfo)
	{
		/* Compute HOF descriptor */
		cv::Mat flows[2];
		cv::split(flow, flows);
		version_1::BuildDescMat(flows[0], flows[1], desc, descInfo);
	}

	void GetDesc(const DescMat &descMat, const DescInfo &descInfo, std::vector<double> &desc)
	{
		/* get a descriptor from the integral histogram. Each histogram is normalized by root-L1 */
		int dim = descInfo.dims, nBins = descInfo.nBins, height = descMat.height, width = descMat.width;
		int xStride = descInfo.width / descInfo.nxCells, yStride = descInfo.height / descInfo.nyCells;
		int xStep = xStride * nBins, yStep = yStride * width * nBins;

		int pos = 0;
		for (int rx = 0; rx < width - descInfo.width; rx += descInfo.width)
			for (int ry = 0; ry < height - descInfo.height; ry += descInfo.height)
			{
				/* iterate over different cells */
				int iDesc = 0;
				std::vector<float> vec(dim, (float)0.);
				float sum = (float)0.;

				for (int xPos = rx, x = 0; x != descInfo.nxCells; xPos += xStride, ++x)
					for (int yPos = ry, y = 0; y != descInfo.nyCells; yPos += yStride, ++y)
					{
						/* get the positions in the integral histogram */
						const float *top_left = descMat.desc + (yPos * width + xPos) * nBins;
						const float *top_right = top_left + xStep;
						const float *bottom_left = top_left + yStep;
						const float *bottom_right = bottom_left + xStep;

						for (int i = 0; i != nBins; ++i, ++iDesc)
						{
							sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
							vec[iDesc] = std::max<float>(sum, 0) + 0.05f;
						}
					}

				float norm = 0.f;
				for (int i = 0; i != dim; ++i)
					norm += vec[i];
				if (norm > 0.f) norm = 1.f / norm;

				for (int i = 0; i != dim; ++i, ++pos) // normalization
					desc[pos] = (double)vec[i] * double(norm);
			}

	}
};

void GetRect(const cv::Point &point, cv::Rect &rect, const int width, const int height, const DescInfo & descInfo)
{
	int x_min = descInfo.width / 2, y_min = descInfo.height / 2;
	int x_max = width - descInfo.width, y_max = height - descInfo.height;

	rect.x = std::min<int>(std::max<int>(cvRound(point.x) - x_min, 0), x_max);
	rect.y = std::min<int>(std::max<int>(cvRound(point.y) - y_min, 0), y_max);
	rect.width = descInfo.width;
	rect.height = descInfo.height;
}

void BuildDescMat(const cv::Mat &xComp, const cv::Mat &yComp, float *desc, const DescInfo &descInfo)
{
	/* compute integral histograms for the whole image */
	float maxAngle = 360.f;
	int nDims = descInfo.nBins;

	/* one more bin for hof */
	int nBins = descInfo.nBins - 1;
	const float angleBase = float(nBins) / maxAngle;

	int step = (xComp.cols + 1) * nDims;
	int index = step + nDims;
	for (int i = 0; i != xComp.rows; ++i, index += nDims)
	{
		const float* xc = xComp.ptr<float>(i);
		const float* yc = yComp.ptr<float>(i);

		/* summarization of the current line */
		std::vector<float> sum(nDims);
		for (int j = 0; j != xComp.cols; ++j)
		{
			float x = xc[j], y = yc[j];
			float mag0 = sqrt(x*x + y*y), mag1;
			int bin0, bin1;

			/* for the zero bin of hof */
			if (mag0 <= min_flow)
			{
				bin0 = nBins;	// the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else
			{
				float angle = cv::fastAtan2(y, x);
				if (angle >= maxAngle) angle -= maxAngle;

				/* split the mag to two adjacent bins */
				float fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0 + 1) % nBins;

				mag1 = (fbin - bin0) * mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for (int m = 0; m != nDims; ++m, ++index)
				desc[index] = desc[index - step] + sum[m];
		}
	}
}

void GetDesc(const DescMat &descMat, const cv::Rect &rect, const DescInfo &descInfo, std::vector<float> &desc, const int index)
{
	/* get a descriptor from the integral histogram. Each histogram is normalized by root-L1 */
	int dim = descInfo.dims, nBins = descInfo.nBins, height = descMat.height, width = descMat.width;
	int xStride = rect.width / descInfo.nxCells, yStride = rect.height / descInfo.nyCells;
	int xStep = xStride * nBins, yStep = yStride * width * nBins;

	/* iterate over different cells */
	int iDesc = 0;
	std::vector<float> vec(dim, (float)0.);
	float sum = (float)0.;
	for (int xPos = rect.x, x = 0; x != descInfo.nxCells; xPos += xStride, ++x)
		for (int yPos = rect.y, y = 0; y != descInfo.nyCells; yPos += yStride, ++y)
		{
			/* get the positions in the integral histogram */
			const float *top_left = descMat.desc + (yPos * width + xPos) * nBins;
			const float *top_right = top_left + xStep;
			const float *bottom_left = top_left + yStep;
			const float *bottom_right = bottom_left + xStep;

			for (int i = 0; i != nBins; ++i, ++iDesc)
			{
				sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
				vec[iDesc] = std::max<float>(sum, 0) + epsilon;
			}
		}

	float norm = 0.f;
	for (int i = 0; i != dim; ++i)
		norm += vec[i];
	if (norm > 0.f) norm = 1.f / norm;

	int pos = 0;
	for (int i = 0; i != dim; ++i, ++pos) // normalization
		desc[pos] += vec[i] * norm;
}

int ComputeDesc(const DescMat &descMat, const DescInfo &descInfo, std::vector<double> &desc)
{
	/* get a descriptor from the integral histogram. Each histogram is normalized by root-L1 */
	int dim = descInfo.dims, nBins = descInfo.nBins, height = descMat.height, width = descMat.width;
	int xStride = descInfo.width / descInfo.nxCells, yStride = descInfo.height / descInfo.nyCells;
	int xStep = xStride * nBins, yStep = yStride * width * nBins;

	int pos = 0;
    int blks = (descMat.width / descInfo.width) * (descMat.height / descInfo.height);
	desc.resize(blks * descInfo.dims);

	for (int rx = 0; rx < width - descInfo.width; rx += descInfo.width)
		for (int ry = 0; ry < height - descInfo.height; ry += descInfo.height)
		{
			/* iterate over different cells */
			int iDesc = 0;
			std::vector<float> vec(dim, (float)0.);
			float sum = (float)0.;

			for (int xPos = rx, x = 0; x != descInfo.nxCells; xPos += xStride, ++x)
				for (int yPos = ry, y = 0; y != descInfo.nyCells; yPos += yStride, ++y)
				{
					/* get the positions in the integral histogram */
					const float *top_left = descMat.desc + (yPos * width + xPos) * nBins;
					const float *top_right = top_left + xStep;
					const float *bottom_left = top_left + yStep;
					const float *bottom_right = bottom_left + xStep;

					for (int i = 0; i != nBins; ++i, ++iDesc)
					{
						sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
						vec[iDesc] = std::max<float>(sum, 0) + epsilon;
					}
				}

			double norm = 0.;
			for (int i = 0; i != dim; ++i)
				norm += (double)vec[i];
			if (norm > 0.) norm = 1. / norm;

			for (int i = 0; i != dim; ++i, ++pos) // normalization
				desc[pos] += double(vec[i]) * norm;
		}

    return blks;
}

void GetVideos(const std::string &fileName, std::vector<cv::Mat> &frames,
	const unsigned int &width, const unsigned int &height, const unsigned int &num, const unsigned int &fps)
{
	/* Check filename to descide the open mode */
	size_t nn = fileName.size();

	if ((fileName[nn - 1] == 'v' || fileName[nn - 1] == 'V')
		&& (fileName[nn - 2] == 'u' || fileName[nn - 2] == 'U')
		&& (fileName[nn - 3] == 'y' || fileName[nn - 3] == 'Y'))
		ReadYUV420p(fileName, frames, width, height, num, fps);
	else
		ReadCommonVideos(fileName, frames, num);

}

void ReadCommonVideos(const std::string &fileName, std::vector<cv::Mat> &frames, const unsigned int &num)
{
	cv::VideoCapture v(fileName);
	assert(v.isOpened());

	double fps = v.get(CV_CAP_PROP_FPS);
	int width = (int)v.get(CV_CAP_PROP_FRAME_WIDTH),
		height = (int)v.get(CV_CAP_PROP_FRAME_HEIGHT);
	unsigned int nof = (unsigned int)v.get(CV_CAP_PROP_FRAME_COUNT);

	assert(nof >= num);

	int step_t = cvRound(fps / 25.);
	step_t = 1;
	int step_s = 1;
	int minW = std::min<int>(width, height);
	while (minW / step_s > 256)
		++step_s;

	int w = width / step_s, h = height / step_s;
	int noof = 1 + (num - 1) / step_t;

	frames.reserve(noof);

	float norm = float(step_s * step_s), sum = 0.f;

	cv::Mat temp(h, w, CV_32FC1, cv::Scalar(0)), frame_c(height, width, CV_8UC3, cv::Scalar(0)), frame_yuv(height, width, CV_8UC3, cv::Scalar(0)), frame_comp[3];

	int row, col, i, j;
	row = col = i = j = 0;
	for (unsigned int nof = 0; nof < num; ++nof)
	{

		v >> frame_c;
		cv::cvtColor(frame_c, frame_yuv, CV_BGR2YUV); // This converting will produce different results between debug and release models.
		cv::split(frame_yuv, frame_comp);

		uchar *buf = reinterpret_cast<uchar *>(frame_comp[0].data);

		for (row = 0; row != h; ++row)
		{
			float *ptr = temp.ptr<float>(row);
			for (col = 0; col != w; ++col)
			{
				sum = 0.f;
				for (i = 0; i != step_s; ++i)
					for (j = 0; j != step_s; ++j)
						sum += (float)buf[(row * step_s + j) * width + step_s * col + i];
				ptr[col] = sum / norm;
			}
		}
		frames.push_back(temp.clone());
	}

	v.release();
}


void ReadYUV420p(const std::string &fileName, std::vector<cv::Mat> &frames,
	const unsigned int &width, const unsigned int &height, const unsigned int &num, const unsigned int &fps)
{
	int step_t = cvRound(float(fps) / 25.f);
	step_t = 1;
	int step_s = 1;
	int minW = std::min<int>(width, height);
	while (minW / step_s > 256)
		++step_s;

	int w = width / step_s, h = height / step_s;
	int noof = 1 + (num - 1) / step_t;

	frames.reserve(noof);

	std::ifstream Input(fileName, std::ios::in | std::ios::binary);
	assert(Input.is_open());

	uchar *buffer = new uchar[width * height];
	float norm = float(step_s * step_s), sum = 0.f;
	float *ptr;
	cv::Mat temp(h, w, CV_32FC1);

	int row, col, i, j;
	for (unsigned int nof = 0; nof < num; ++nof)
	{
		Input.seekg(nof * width * height / 2 * 3, Input.beg);
		Input.read(reinterpret_cast<char *>(buffer), width * height);

		for (row = 0; row != h; ++row)
		{
			ptr = temp.ptr<float>(row);
			for (col = 0; col != w; ++col)
			{
				sum = 0.f;
				for (i = 0; i != step_s; ++i)
					for (j = 0; j != step_s; ++j)
						sum += (float)buffer[(row * step_s + j) * width + step_s * col + i];
				ptr[col] = sum / norm;
			}
		}
		frames.push_back(temp.clone());
	}
	ptr = nullptr;
	Input.close();
	delete(buffer);

}

void GetOrigYUV420p(const std::string &fileName, std::vector<cv::Mat> &frames,
	const unsigned int &width, const unsigned int &height, const unsigned int &noof, const unsigned int &fps)
{
	int w = width, h = height;

	frames.reserve(noof);

	std::ifstream Input(fileName, std::ios::in | std::ios::binary);
	assert(Input.is_open());

	uchar *buffer = new uchar[width * height];
	cv::Mat temp(h, w, CV_32FC1);

	for (unsigned int nof = 0; nof < noof; ++nof)
	{
		Input.seekg(nof * width * height / 2 * 3, Input.beg);
		Input.read(reinterpret_cast<char *>(buffer), width * height);
		cv::Mat(height, width, CV_8UC1, buffer).convertTo(temp, CV_32FC1);
		frames.push_back(temp.clone());
	}
	Input.close();
	delete(buffer);
}

void ComputeSimpleGradient(const cv::Mat &img, cv::Mat &xComp, cv::Mat &yComp)
{
	xComp.create(img.size(), CV_32FC1);
	yComp.create(img.size(), CV_32FC1);

	cv::Mat im(img.size(), CV_32FC1);
	img.convertTo(im, CV_32FC1);

	cv::Mat dx(3, 3, CV_32FC1), dy(3, 3, CV_32FC1);
	float *ptr = (float*)dx.data;
	ptr[0] = 1.f / 3.f; ptr[1] = 1.f / 3.f; ptr[2] = 1.f / 3.f;
	ptr[3] = 0.f; ptr[4] = 0.f; ptr[5] = 0.f;
	ptr[6] = -1.f / 3.f; ptr[7] = -1.f / 3.f; ptr[8] = -1.f / 3.f;

	cv::transpose(dx, dy);

	cv::filter2D(im, xComp, CV_32FC1, dx, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(im, yComp, CV_32FC1, dy, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
}

void HofComp(const cv::Mat &flow, float *desc, const DescInfo &descInfo)
{
	/* Compute HOF descriptor */
	cv::Mat flows[2];
	cv::split(flow, flows);
	BuildDescMat(flows[0], flows[1], desc, descInfo);
}

void ComputeGradient(const cv::Mat &img, cv::Mat &gradient)
{
	cv::Mat imgX, imgY;

	ComputeSimpleGradient(img, imgX, imgY);
	cv::magnitude(imgX, imgY, gradient);
}

void ComputeGMSM(const cv::Mat &org, const cv::Mat &dst, cv::Mat &grad_chg)
{
	cv::Mat grad_org(org.size(), CV_32FC1), grad_dst(org.size(), CV_32FC1);
	ComputeGradient(org, grad_org);
	ComputeGradient(dst, grad_dst);

	cv::multiply(grad_org, grad_dst, grad_chg, 2.);
	grad_chg += arg_c;
	cv::pow(grad_org, 2, grad_org); cv::pow(grad_dst, 2, grad_dst);
	cv::divide(grad_chg, grad_org + grad_dst + arg_c, grad_chg);
}

void ComputeGMS(const cv::Mat &org, const cv::Mat &dst, std::vector<cv::Mat> &chg)
{
	assert(org.size() == dst.size());

	cv::Mat grad_chg(org.size(), CV_32FC1);
	ComputeGMSM(org, dst, grad_chg);
	chg.push_back(grad_chg.clone());
}

void ComputeGMSD(const cv::Mat &org, const cv::Mat &dst, std::vector<double> &quality, const int &num)
{
	assert(org.size() == dst.size());

	cv::Mat grad_chg(org.size(), CV_32FC1);
	cv::Mat meanMat, stdMat;

	ComputeGMSM(org, dst, grad_chg);
	cv::meanStdDev(grad_chg, meanMat, stdMat);
	quality[num] = stdMat.ptr<double>(0)[0];

}

void ComputeGMSD(const std::vector<cv::Mat> &org, const std::vector<cv::Mat> &dst, std::vector<std::vector<double>> &quality, const int &num)
{
	assert(org.size() == dst.size());
	assert(org[0].size() == dst[0].size());

	for (size_t i = 0; i != org.size(); ++i)
	{
		cv::Mat grad_chg(org[i].size(), CV_32FC1);
		cv::Mat meanMat, stdMat;

		ComputeGMSM(org[i], dst[i], grad_chg);
		cv::meanStdDev(grad_chg, meanMat, stdMat);
		quality[i][num] = stdMat.ptr<double>(0)[0];
	}
}

void CenterBias(cv::Mat &mask, const double &maxVal, const double &minVal)
{
	assert(!mask.empty());

	cv::Mat X(mask.size(), CV_32FC1), Y(mask.size(), CV_32FC1);
	int width = mask.cols, height = mask.rows;

	float cw = (float)width / 2.f;
	float ch = (float)height / 2.f;

	int i, j;
	for (i = 0; i != width; ++i)
		X.col(i) = (float)i - cw;
	for (j = 0; j != height; ++j)
		Y.row(j) = (float)j - ch;

	cv::magnitude(X, Y, mask);
	mask *= -1.f;

	cv::normalize(mask, mask, 1., 0., CV_MINMAX);
	cv::threshold(mask, mask, maxVal, maxVal, cv::THRESH_TRUNC);
	mask *= -1.f;
	cv::threshold(mask, mask, -minVal, -minVal, cv::THRESH_TRUNC);
	mask *= -1.f;
	cv::normalize(mask, mask, 1., 0., CV_MINMAX);

	return;
}

void Compute3DGM(const std::vector<cv::Mat> &frames, cv::Mat &gm, cv::Mat &xComp, cv::Mat &yComp, cv::Mat &tComp, const size_t &num)
{
	// ONLY compute simple gradient in 3-D 
	assert(num > 0);

	cv::Mat temp1, temp2;

	// dx = [1 0 -1] --> xComp;
	float x_ptr[9] = { 1.f, 0.f, -1.f, 1.f, 0.f, -1.f, 1.f, 0.f, -1.f };
	cv::Mat dx(3, 3, CV_32FC1, x_ptr);
	cv::filter2D(frames[num - 1], xComp, -1, dx, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num], temp1, -1, dx, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num + 1], temp2, -1, dx, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	xComp += (temp1 + temp2);
	xComp /= 9.f;

	// dy = [1; 0; -1] --> yComp;
	float y_ptr[9] = { 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, -1.f, -1.f, -1.f };
	cv::Mat dy(3, 3, CV_32FC1, y_ptr);
	cv::filter2D(frames[num - 1], yComp, -1, dy, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num], temp1, -1, dy, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num + 1], temp2, -1, dy, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	yComp += (temp1 + temp2);
	yComp /= 9.f;

	// dt = [1 1 1][0 0 0][-1 -1 -1] --> tComp
	cv::boxFilter(frames[num - 1], tComp, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	cv::boxFilter(frames[num + 1], temp1, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	tComp -= temp1;

	// gradient magnitude
	cv::pow(xComp, 2, temp1);
	cv::pow(yComp, 2, temp2);
	cv::pow(tComp, 2, gm);
	gm += (temp1 + temp2);
	cv::sqrt(gm, gm);

	return;
}

void Compute3DGM_Sobel(const std::vector<cv::Mat> &frames, cv::Mat &gm, cv::Mat &xComp, cv::Mat &yComp, cv::Mat &tComp, const size_t &num)
{
	// ONLY compute simple gradient in 3-D 
	assert(num > 0);

	cv::Mat temp1, temp2;

	// dx = [1 0 -1] --> xComp;
	float x_ptr_1[9] = { 1.f, 0.f, -1.f, 3.f, 0.f, -3.f, 1.f, 0.f, -1.f };
	float x_ptr_2[9] = { 3.f, 0.f, -3.f, 6.f, 0.f, -6.f, 3.f, 0.f, -3.f };
	cv::Mat dx_1(3, 3, CV_32FC1, x_ptr_1);
	cv::Mat dx_2(3, 3, CV_32FC1, x_ptr_2);
	cv::filter2D(frames[num - 1], xComp, -1, dx_1, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num], temp1, -1, dx_2, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num + 1], temp2, -1, dx_1, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	xComp += (temp1 + temp2);

	// dy = [1; 0; -1] --> yComp;
	float y_ptr_1[9] = { 1.f, 3.f, 1.f, 0.f, 0.f, 0.f, -1.f, -3.f, -1.f };
	float y_ptr_2[9] = { 3.f, 6.f, 3.f, 0.f, 0.f, 0.f, -3.f, -6.f, -3.f };
	cv::Mat dy_1(3, 3, CV_32FC1, y_ptr_1);
	cv::Mat dy_2(3, 3, CV_32FC1, y_ptr_2);
	cv::filter2D(frames[num - 1], yComp, -1, dy_1, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num], temp1, -1, dy_2, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num + 1], temp2, -1, dy_1, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	yComp += (temp1 + temp2);

	// dt = [1 1 1][0 0 0][-1 -1 -1] --> tComp
	float t_ptr[9] = { 1.f, 3.f, 1.f, 3.f, 6.f, 3.f, 1.f, 3.f, 1.f };
	cv::Mat dt(3, 3, CV_32FC1, t_ptr);
	cv::filter2D(frames[num - 1], tComp, -1, dt, cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	cv::filter2D(frames[num + 1], temp1, -1, dt, cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	tComp -= temp1;

	// gradient magnitude
	cv::pow(xComp, 2, temp1);
	cv::pow(yComp, 2, temp2);
	cv::pow(tComp, 2, gm);
	gm += (temp1 + temp2);
	cv::sqrt(gm, gm);

	return;
}

void DenseSample(const cv::Mat& frame, const cv::Mat &mask, std::list<Track> &track, const TrackInfo &trackInfo, const DescInfo &hofInfo, const int &sample_step, const int &frame_num)
{
	/* detect new feature points in an image without overlapping to previous points */
	int sample_width, sample_height;
	int x_max, y_max;
	int x, y, x_, y_;
	cv::Point point;

	sample_width = frame.cols / sample_step;
	sample_height = frame.rows / sample_step;

	cv::Mat eig;
	cv::cornerMinEigenVal(frame, eig, eig_size, eig_size);
	cv::multiply(eig, mask, eig);

	double maxVal = 0;
	cv::minMaxLoc(eig, 0, &maxVal);
	double threshold = maxVal * sample_quality;

	std::vector<int> counters(sample_width * sample_height);
	x_max = sample_step * sample_width;
	y_max = sample_step * sample_height;

	/* put previous points */
	for (auto &pList : track)
	{
		point = pList.point[pList.index];
		x = point.x;
		y = point.y;

		if (x < x_max && y < y_max)
		{
			x /= sample_step;
			y /= sample_step;
			++counters[y * sample_width + x];
		}
	}

	int index = 0;
	int offset = sample_step / 2;
	float *ptr;
	for (int i = 0; i != sample_height; ++i)
	{
		y_ = i * sample_step + offset;
		ptr = eig.ptr<float>(y_);
		for (int j = 0; j != sample_width; ++j, ++index)
		{
			if (counters[index] > 0) // no overlappiing points
				continue;

			x_ = j * sample_step + offset;

			if (ptr[x_] > threshold) // points should have a large eigen val
			{
				track.push_back(Track(cv::Point(x_, y_), trackInfo, hofInfo, frame_num));
			}
		}
	}

	ptr = nullptr;
}

void ForegroundSampling(const cv::Mat& frame, const cv::Mat &saliencyMask,
	std::list<Track> &track, const TrackInfo &trackInfo, const DescInfo &hofInfo, const int &sample_step, const int &frame_num)
{
	/* detect new feature points in an image without overlapping to previous points */
	int sample_width, sample_height;
	int x_max, y_max;
	int x, y, x_, y_;
	cv::Point point;

	sample_width = frame.cols / sample_step;
	sample_height = frame.rows / sample_step;

	/*
	cv::Mat eig;
	cv::cornerMinEigenVal(frame, eig, eig_size, eig_size);
	cv::threshold(eig, eig, 1000, 1000, cv::THRESH_TRUNC);
	cv::multiply(eig, saliencyMask, eig);

	double maxVal = 0;
	cv::minMaxLoc(eig, 0, &maxVal);
	double threshold = maxVal * sample_quality;
	*/
	double threshold = 0.15;

	std::vector<int> counters(sample_width * sample_height);
	x_max = sample_step * sample_width;
	y_max = sample_step * sample_height;

	/* put previous points */
	for (auto &pList : track)
	{
		point = pList.point[pList.index];
		x = point.x;
		y = point.y;

		if (x < x_max && y < y_max)
		{
			x /= sample_step;
			y /= sample_step;
			++counters[y * sample_width + x];
		}
	}

	/* -------------------- */
	//cv::Mat temp;
	//frame.copyTo(temp);
	/* -------------------- */

	cv::Mat saliency;
	saliencyMask.copyTo(saliency);

	int index = 0;
	int offset = sample_step / 2;
	float *ptr;
	for (int i = 0; i != sample_height; ++i)
	{
		y_ = i * sample_step + offset;
		//ptr = eig.ptr<float>(y_);
		ptr = saliency.ptr<float>(y_);

		for (int j = 0; j != sample_width; ++j, ++index)
		{
			if (counters[index] > 0) // no overlappiing points
				continue;

			x_ = j * sample_step + offset;

			if (ptr[x_] > threshold) // points should have a large eigen val
			{
				track.push_back(Track(cv::Point(x_, y_), trackInfo, hofInfo, frame_num));

				/* ---------------- */
				//cv::rectangle(temp, cv::Rect(std::max<int>(x_ - 24, 0), std::max<int>(y_ - 24, 0), 48, 48), cv::Scalar(255, 0, 255), 2);
				/* ---------------- */
			}
		}
	}

	ptr = nullptr;
}

bool IsValid(const std::vector<cv::Point> &trajectory, const std::vector<cv::Point2f> &displacement)
{
	float mean_x = 0.f, mean_y = 0.f, var_x = 0.f, var_y = 0.f;
	int sizes = (int)trajectory.size();
	float norm = 1.f / sizes;
	for (int i = 0; i != sizes; ++i)
	{
		mean_x += (float)trajectory[i].x;
		mean_y += (float)trajectory[i].y;
	}

	mean_x *= norm; mean_y *= norm;

	float temp_x, temp_y;
	for (int i = 0; i != sizes; ++i)
	{
		temp_x = (float)trajectory[i].x - mean_x;
		temp_y = (float)trajectory[i].y - mean_y;
		var_x += (temp_x * temp_x);
		var_y += (temp_y * temp_y);
	}
	var_x *= norm; var_y *= norm;
	var_x = sqrt(var_x); var_y = sqrt(var_y);

	// remove random trajectory.
	if (var_x > max_var || var_y > max_var)
		return false;

	float disp_max = 0., disp_sum = 0.;
	float temp = 0.;
	for (int i = 0; i != displacement.size(); ++i)
	{
		temp = (float)cv::norm(displacement[i]);

		disp_sum += temp;
		if (disp_max < temp)
			disp_max = temp;
	}

	// remove random step.
	if ((disp_max > max_dis) && (disp_max > disp_sum*0.7))
		return false;

	//remove static trajectory.
	if (float(static_thre) < 0.00001f) // if the threshold is small enough, we guess you do not want to remove the static trajectories.
		return true;
	else if (disp_sum < static_thre)
		return false;

	return true;
}

bool IsOverlapping(const std::vector<cv::Point> &traj, std::list<std::vector<cv::Point>> &region_inhibit, const int &len)
{
	float max_inhibit_dist = float(len) * float(patch_size) * inhibit_quality;
	for (auto iter = region_inhibit.cbegin(); iter != region_inhibit.cend(); ++iter)
	{
		float max_distance = 0., sum_distance = 0.;
		for (int j = 0; j != len; ++j)
		{
			float temp = (float)cv::norm(traj[j] - (*iter)[j]);
			max_distance = (temp > max_distance ? temp : max_distance);
			sum_distance += temp;
		}

		if (sum_distance < max_inhibit_dist)
			return false;

		if ((sum_distance > max_inhibit_dist) && (max_distance > (sum_distance * 0.7)))
			return false;
	}

	region_inhibit.push_back(traj);
	return true;
}

void CalcFlowWithFarneback(const cv::Mat &prev, const cv::Mat &curr, cv::Mat &flow0, const FlowInfo &flowInfo)
{
	/* optical flow calculation */
	flow0.create(prev.size(), CV_32FC2);

	int i, k;
	double scale = 1.;
	cv::Mat prevFlow, flow;
	int levels = flowInfo.levels;

	std::vector<cv::Mat> pyrPrev(levels + 1), pyrCurr(levels + 1);
	prev.copyTo(pyrPrev[0]);
	curr.copyTo(pyrCurr[0]);

	for (size_t l = 1; l != pyrPrev.size(); ++l)
	{
		scale *= flowInfo.scale;
		cv::resize(pyrPrev[0], pyrPrev[l], cv::Size(), scale, scale, CV_INTER_LINEAR);
		cv::resize(pyrCurr[0], pyrCurr[l], cv::Size(), scale, scale, CV_INTER_LINEAR);
	}

	for (k = levels; k >= 0; k--)
	{
		int width = pyrPrev[k].cols;
		int height = pyrPrev[k].rows;;

		if (k > 0)
			flow.create(height, width, CV_32FC2);
		else
			flow = flow0;

		if (prevFlow.empty())
		{
			flow = cv::Mat::zeros(height, width, CV_32FC2);
		}
		else
		{
			resize(prevFlow, flow, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
			flow *= 1. / flowInfo.scale;
		}

		cv::Mat R[2], M;
		Farneback::FarnebackPolyExp(pyrPrev[k], R[0], flowInfo.poly_n, flowInfo.poly_sigma);
		Farneback::FarnebackPolyExp(pyrCurr[k], R[1], flowInfo.poly_n, flowInfo.poly_sigma);


		Farneback::FarnebackUpdateMatrices(R[0], R[1], flow, M, 0, flow.rows);

		for (i = 0; i < flowInfo.iteration; ++i)
		{
			Farneback::FarnebackUpdateFlow_Blur(R[0], R[1], flow, M, flowInfo.winSize, i < flowInfo.iteration - 1);
		}

		prevFlow = flow;
	}
}

void ComputeDissimilarity(const std::vector<float> &vec1, const std::vector<float> &vec2, std::list<double> &tQuality)
{
	assert(vec1.size() == vec2.size());
	if (vec1.size() == 0)
		return;
	std::vector<double> dis(vec1.size(), 0.);

	int ndims = (int)vec1.size();
	std::transform(vec1.begin(), vec1.end(), vec2.begin(), dis.begin(),
		[](const float &x, const float &y){return (2 * (double)x*(double)y + arg_T) / ((double)x*(double)x + (double)y*(double)y + arg_T); });

	double sum = 0.;
	for (int i = 0; i != ndims; ++i)
		sum += dis[i];
	sum /= double(ndims);

	tQuality.push_back(sum);
}

void ComputeDissimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2, std::list<double> &tQuality)
{
	assert(vec1.size() == vec2.size());
	if (vec1.size() == 0)
		return;
	std::vector<double> dis(vec1.size(), 0.);

	int ndims = (int)vec1.size();
	std::transform(vec1.begin(), vec1.end(), vec2.begin(), dis.begin(),
		[](const double &x, const double &y){return (2 * x*y + arg_T) / (x*x + y*y + arg_T); });

	double sum = 0.;
	for (int i = 0; i != ndims; ++i)
		sum += dis[i];
	sum /= double(ndims);

	tQuality.push_back(sum);
}

double GetMeanVal(const std::list<double> &lst)
{
	double sum = 0.;
	int ndims = (int)lst.size();

	if (ndims == 0)
		return 0.;
	for (auto iter = lst.begin(); iter != lst.end(); ++iter)
	{
		sum += *iter;
	}

	return (sum / double(ndims));
}

double GetMeanVal(const std::vector<double> &vec)
{
	double sum = 0.;
	int ndims = (int)vec.size();
	for (auto iter = vec.begin(); iter != vec.end(); ++iter)
	{
		sum += *iter;
	}

	return (sum / double(ndims));
}

double SimpleTemporalPooling(const std::list<double> &tq)
{
	double meanVal = 0., stdev = 0.;

	double q = 0.;
	double dims = (double)tq.size();
	if (dims > 1.9)
	{
		for (const double &elem : tq)
			meanVal += elem;
		meanVal /= dims;

		for (const double &elem : tq)
			stdev += ((meanVal - elem) * (meanVal - elem));
		stdev = sqrt(stdev / (dims - 1.));

		q = meanVal + stdev;
	}
	else
		q = GetMeanVal(tq);
	return q;
}

double SimpleTemporalPooling(const std::vector<double> &tq)
{
	double meanVal = 0., stdev = 0.;

	double q = 0.;
	double dims = (double)tq.size();
	if (dims > 1.9)
	{
		for (const double &elem : tq)
			meanVal += elem;
		meanVal /= dims;

		for (const double &elem : tq)
			stdev += ((meanVal - elem) * (meanVal - elem));
		stdev = sqrt(stdev / (dims - 1.));

		q = meanVal + stdev;
	}
	else
		q = GetMeanVal(tq);
	return q;
}

void GetCube(const std::vector<cv::Mat> &frames, std::vector<cv::Mat> &cube, const std::vector<cv::Rect> &rects, const int &start_, const int &end_)
{
	int j = 0;
	for (int i = start_; i != end_; ++i)
	{
		j = i - start_;
		cube.push_back(frames[i](rects[j]).clone());
	}

	return;
}

void GetScore_t(const std::vector<float>& desc_org, const std::vector<float>& desc_dst,
	const DescInfo &descInfo, const TrackInfo &trackInfo, std::list<double> &tQuality)
{
	/* method 1 -- mapping all histograms to a single 2*2*8 based histogram */

	assert(desc_org.size() == desc_dst.size());
	int ndims = (int)desc_org.size();
	if (ndims == 0)
		return;
	std::vector<double> dis(ndims, 0.);

	/* (2*x*y + T) / (x*x + y*y + T) */
	std::transform(desc_org.begin(), desc_org.end(), desc_dst.begin(), dis.begin(),
		[](const float &x, const float &y){return 1. - (2 * (double)x*(double)y + arg_T) / ((double)x*(double)x + (double)y*(double)y + arg_T); });

	double mv = 0., stdev = 0.;
	for (double &elem : dis)
		mv += elem;

	mv /= double(ndims);
	tQuality.push_back(mv);
}

void GetScore_s(std::list<double> &sQuality, const std::vector<double> &sFrame, const int &start, const int &end)
{
	double sum = 0.0;
	for (int i = start; i != end; ++i)
		sum += sFrame[i];
	double num = end - start;

	double meanVal = sum / num;
	sQuality.push_back(meanVal);
}

void GetScore_st(const std::vector<cv::Mat> &OrigCube, const std::vector<cv::Mat> &DistCube, std::list<double> &tST)
{
	cv::Mat gmOrig, gmDist, xComp, yComp, tComp;
	size_t sizes = OrigCube.size(), frame_num;
	int ww = int(OrigCube[0].rows);
	cv::Mat tempMat(ww * int(OrigCube.size() - 2), ww, CV_32FC1);

	for (frame_num = 1; frame_num < sizes - 1; ++frame_num)
	{
		Compute3DGM(OrigCube, gmOrig, xComp, yComp, tComp, frame_num);
		Compute3DGM(DistCube, gmDist, xComp, yComp, tComp, frame_num);

		cv::Mat grad_chg;
		cv::multiply(gmOrig, gmDist, grad_chg, 2.);
		grad_chg += arg_c;
		cv::pow(gmOrig, 2, gmOrig); cv::pow(gmDist, 2, gmDist);
		cv::divide(grad_chg, gmOrig + gmDist + arg_c, grad_chg);

		grad_chg.copyTo(tempMat(cv::Rect(0, (int(frame_num) - 1)*ww, ww, ww)));
	}

	cv::Mat meanMat, stdMat;
	cv::meanStdDev(tempMat, meanMat, stdMat);
	tST.push_back(stdMat.ptr<double>(0)[0]);
}

void ComputeGMSD_3D(const std::vector<cv::Mat> &org, const std::vector<cv::Mat> &dst, std::vector<double> &quality, const int &num)
{
	cv::Mat gmOrig, gmDist, xComp, yComp, tComp;

	Compute3DGM(org, gmOrig, xComp, yComp, tComp, num+1);
	Compute3DGM(dst, gmDist, xComp, yComp, tComp, num+1);

	cv::Mat grad_chg;
	cv::multiply(gmOrig, gmDist, grad_chg, 2.);
	grad_chg += arg_c;
	cv::pow(gmOrig, 2, gmOrig); cv::pow(gmDist, 2, gmDist);
	cv::divide(grad_chg, gmOrig + gmDist + arg_c, grad_chg);

	cv::Mat meanMat, stdMat;
	cv::meanStdDev(grad_chg, meanMat, stdMat);
	quality.push_back(stdMat.ptr<double>(0)[0]);
}

//////////////////////////////////////////////////////////////////////////
void run_build(const cv::Mat &xComp, const cv::Mat &yComp, float *desc)
{
	/* compute integral histograms for the whole image */
	float maxAngle = 360.f;
	int nDims = 8;

	/* one more bin for hof */
	int nBins = nDims - 1;
	const float angleBase = float(nBins) / maxAngle;

	int step = (xComp.cols + 1) * nDims;
	int index = step + nDims;
	for (int i = 0; i != xComp.rows; ++i, index += nDims)
	{
		const float* xc = xComp.ptr<float>(i);
		const float* yc = yComp.ptr<float>(i);

		/* summarization of the current line */
		std::vector<float> sum(nDims);
		for (int j = 0; j != xComp.cols; ++j)
		{
			float x = xc[j], y = yc[j];
			float mag0 = sqrt(x*x + y*y), mag1;
			int bin0, bin1;

			/* for the zero bin of hof */
			if (mag0 <= min_flow)
			{
				bin0 = nBins;	// the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else
			{
				float angle = cv::fastAtan2(y, x);
				if (angle >= maxAngle) angle -= maxAngle;

				/* split the mag to two adjacent bins */
				float fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0 + 1) % nBins;

				mag1 = (fbin - bin0) * mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for (int m = 0; m != nDims; ++m, ++index)
				desc[index] = desc[index - step] + sum[m];
		}
	}
}

void run_grid_desc(const float *desc_In, const int width, const int height, std::vector<double> &desc_Out)
{
	int grid = 48;
	int nCells = 2;
	int bins = 8;

	int nW = width / grid;
	int nH = height / grid;
	int w_off = (width - nW * grid) / 2;
	int h_off = (height - nH * grid) / 2;

	desc_Out.resize(nW*nH*nCells*nCells*bins);

	/* get a descriptor from the integral histogram. Each histogram is normalized by root-L1 */
	int dim = nCells * nCells * bins, nBins = bins;
	int xStride = grid / nCells, yStride = grid / nCells;
	int xStep = xStride * nBins, yStep = yStride * (width+1) * nBins;
    
	int pos = 0;
	for (int rx = w_off; rx < width - grid + 1; rx += grid)
		for (int ry = h_off; ry < height - grid + 1; ry += grid)
		{
			/* iterate over different cells */
			int iDesc = 0;
			std::vector<float> vec(dim, (float)0.);
			float sum = (float)0.;

			for (int xPos = rx, x = 0; x != nCells; xPos += xStride, ++x)
				for (int yPos = ry, y = 0; y != nCells; yPos += yStride, ++y)
				{
					/* get the positions in the integral histogram */
					const float *top_left = desc_In + (yPos * (width + 1) + xPos) * nBins;
					const float *top_right = top_left + xStep;
					const float *bottom_left = top_left + yStep;
					const float *bottom_right = bottom_left + xStep;

					for (int i = 0; i != nBins; ++i, ++iDesc)
					{
						sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
						vec[iDesc] = std::max<float>(sum, 0) + 0.05f;
					}
				}

			float norm = 0.f;
			for (int i = 0; i != dim; ++i)
				norm += vec[i];
			if (norm > 0.f) norm = 1.f / norm;

			for (int i = 0; i != dim; ++i, ++pos) // normalization
				desc_Out[pos] = (double)vec[i] * double(norm);
		}
}

void run(double *ptr_prev, double *ptr_next, const int rows, const int cols, std::vector<double> &desc_out)
{
	cv::Mat prev(cols, rows, CV_64FC1, ptr_prev);
	cv::Mat next(cols, rows, CV_64FC1, ptr_next);

	prev.convertTo(prev, CV_32FC1); prev = prev.t();
	next.convertTo(next, CV_32FC1); next = next.t();

	cv::Mat flow;
	FlowInfo flowInfo(1. / sqrt(2.), 3, 10, 2, 7, 1.5);
	CalcFlowWithFarneback(prev, next, flow, flowInfo);

	int width = flow.cols;
	int height = flow.rows;

	int dims = (width + 1) * (height + 1) * 8;
	float * desc = (float*)malloc(dims * sizeof(float));
	memset(desc, 0, dims * sizeof(float));

	cv::Mat flows[2];
	cv::split(flow, flows);
	run_build(flows[0], flows[1], desc);

	desc_out.clear();
	run_grid_desc(desc, width, height, desc_out);

	free(desc);
}

void detect_kp(double *ptr, cv::Mat &out, const int rows, const int cols)
{
    cv::Mat im(cols, rows, CV_64FC1, ptr);
    
    im.convertTo(im, CV_32FC1); im = im.t();
    
    cv::Mat mask(im.size(), CV_32FC1);
	CenterBias(mask, .9, 0.);
    
    cv::Mat eig;
	cv::cornerMinEigenVal(im, eig, 3, 3);
	cv::multiply(eig, mask, eig);
    
    eig.convertTo(eig, CV_64FC1);
    out = eig.clone();
}