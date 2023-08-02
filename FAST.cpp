#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

#define HEIGHT 720
#define WIDTH 480
// HBITS, WITS are the minimum bits required to represent HEIGHT and WIDTH in unsigned format
#define HBITS 10
#define WBITS 9
// N defines the number of contiguous points not within the threshold.
#define N 9
#define THRESHOLD 20

typedef hls::stream<ap_axis<8,2,5,6>> ISTREAM;
typedef hls::stream<ap_axis<8,2,5,6>> OSTREAM;

ap_axis<8, 2, 5, 6> dat;

void get_image(ISTREAM &in, ap_uint<8> img[HEIGHT][WIDTH]) {
STREAM_READ_L1:for(ap_uint<HBITS> i = 0; i < HEIGHT; i++){
	STREAM_READ_L2:for(ap_uint<WBITS> j = 0; j < WIDTH; j++) {
	#pragma HLS PIPELINE
			in.read(dat);
			img[i][j] = dat.data.to_short();
		}
	}
	dat.last = 0;
}

void check_fp(ap_uint<8> fps[WIDTH], ap_uint<8> img[HEIGHT][WIDTH], const ap_uint<2> tab[512], ap_uint<HBITS> r, ap_uint<WBITS> c) {
#pragma HLS INLINE
	static const ap_int<3> offsets[25][2] =
	    {
	        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
	        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3},
			{0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
			{0, -3}
	    };
	static ap_uint<2> cache[25];
// The cache array stores state of each point on the circle.
// 0 => Absolute difference of point in question and offset point is less than THRESHOLD
// 1 => Difference is negative and more than THRESHOLD
// 2 => Difference is positive and more than THRESHOLD
#pragma HLS ARRAY_PARTITION variable=offsets type=complete
#pragma HLS ARRAY_PARTITION variable=cache type=complete
	cache[0] = tab[img[r][c] - img[r + offsets[0][0]][c + offsets[0][1]] + 255];
	cache[8] = tab[img[r][c] - img[r + offsets[8][0]][c + offsets[8][1]] + 255];
	ap_uint<2> d = cache[0] | cache[8];
	if(!d) {
		fps[c] = 0;
		return;
	}
	cache[12] = tab[img[r][c] - img[r + offsets[12][0]][c + offsets[12][1]] + 255];
	cache[4] = tab[img[r][c] - img[r + offsets[4][0]][c + offsets[4][1]] + 255];

	if(!(cache[4] | cache[12])) {
		fps[c] = 0;
		return;
	}
	cache[2] = tab[img[r][c] - img[r + offsets[2][0]][c + offsets[2][1]] + 255];
	cache[6] = tab[img[r][c] - img[r + offsets[6][0]][c + offsets[6][1]] + 255];
	cache[10] = tab[img[r][c] - img[r + offsets[10][0]][c + offsets[10][1]] + 255];
	cache[14] = tab[img[r][c] - img[r + offsets[14][0]][c + offsets[14][1]] + 255];
	d &= cache[2] | cache[10];
	d &= cache[4] | cache[12];
	d &= cache[6] | cache[14];
	if(!d){
		fps[c] = 0;
		return;
	}

	cache[1] = tab[img[r][c] - img[r + offsets[1][0]][c + offsets[1][1]] + 255];
	cache[3] = tab[img[r][c] - img[r + offsets[3][0]][c + offsets[3][1]] + 255];
	cache[5] = tab[img[r][c] - img[r + offsets[5][0]][c + offsets[5][1]] + 255];
	cache[7] = tab[img[r][c] - img[r + offsets[7][0]][c + offsets[7][1]] + 255];
	cache[9] = tab[img[r][c] - img[r + offsets[9][0]][c + offsets[9][1]] + 255];
	cache[11] = tab[img[r][c] - img[r + offsets[11][0]][c + offsets[11][1]] + 255];
	cache[13] = tab[img[r][c] - img[r + offsets[13][0]][c + offsets[13][1]] + 255];
	cache[15] = tab[img[r][c] - img[r + offsets[15][0]][c + offsets[15][1]] + 255];

	d &= cache[1] | cache[9];
	d &= cache[3] | cache[11];
	d &= cache[5] | cache[13];
	d &= cache[7] | cache[15];


	if(!d) {
		fps[c] = 0;
		return;
	}
//
//	d = 0;
//	d |= tab[img[r][c] - img[r + offsets[4][0]][c + offsets[4][1]]];
//	d |= tab[img[r][c] - img[r + offsets[12][0]][c + offsets[12][1]]];
//	if(!d) {
//		fps[c] = 0;
//		return;
//	}
	d = cache[0];
	ap_uint<4> count = cache[0] ? 1 : 0;

// Fill the remaining points as the we need to check N contiguous points in the circle
	for(ap_uint<5> i = 16; i < 25; i++) {
#pragma HLS UNROLL
		cache[i] = cache[i - 16];
	}
//	ap_uint<4> count = 0;
	for(ap_uint<5> i = 1; i < 25; i++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=25
#pragma HLS UNROLL
//		if(i - count > 25 - N) {
//			fps[c] = 0;
//			return;
//		}

// Here d stores the state of last point (0,1,2)
		if(d and (cache[i] == d)) {
			count++;
		} else if(cache[i]) {
			count = 1;
			d = cache[i];
		}
		if(cache[i] == 0){
			count = 0;
			d = 0;
		}
		if(count == 9) {
			fps[c] = 1;
			return;
		}
	}
	fps[c] = 0;
}

void eval_FAST(OSTREAM &out, ap_uint<8> img[HEIGHT][WIDTH], const ap_uint<2> tab[512]) {
	dat.data = 0;
// FAST does not check for feature points in the first 3 rows hence, we fill first three rows with 0's
	TOP_FILL:for(ap_uint<WBITS+2> i = 0; i < 3 * WIDTH; i++) {
#pragma HLS PIPELINE
		out.write(dat);
	}

	static ap_uint<8> fps[WIDTH] = {0,};
// fps holds the information on all the feature points in row i. 1 -> is a feature point, 0 is not a feature point

	HEIGHT_LOOP:for(ap_uint<HBITS> i = 3; i < HEIGHT - 3; i++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off

#pragma HLS ARRAY_PARTITION variable=fps type=block factor=7
// Note that FAST does not check for feature points in first and last three columns of a row, hence WIDTH - 6
		WIDTH_LOOP:for(ap_uint<WBITS> j = 3; j < (WIDTH - 6) / 6 + 3; j++) {
#pragma HLS PIPELINE II=17
			check_fp(fps, img, tab, i, j);
			check_fp(fps, img, tab, i , j + (WIDTH - 6) / 6);
			check_fp(fps, img, tab, i , j + 2 * ((WIDTH - 6) / 6));
			check_fp(fps, img, tab, i , j + 3 * ((WIDTH - 6) / 6));
			check_fp(fps, img, tab, i , j + 4 * ((WIDTH - 6) / 6));
			check_fp(fps, img, tab, i , j + 5 * ((WIDTH - 6) / 6));
		}

// Write all feature points to the stream
		STREAM_WRITE:for(ap_uint<WBITS> j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE
			dat.data = fps[j];
			out.write(dat);
		}
	}

	dat.data = 0;
// FAST does not check for feature points in bottom three rows
	BOTTOM_FILL:for(ap_uint<WBITS+2> i = 0; i < 3 * WIDTH - 1; i++) {
#pragma HLS PIPELINE
		out.write(dat);
	}
	dat.last = 1;
	out.write(dat);
}

void FAST(ISTREAM &in, OSTREAM &out) {
#pragma HLS INTERFACE port=return mode=s_axilite
#pragma HLS INTERFACE port=in mode=axis
#pragma HLS INTERFACE port=out mode=axis
	ap_uint<8> img[HEIGHT][WIDTH];
#pragma HLS ARRAY_PARTITION variable=img type=block factor=6 dim=2
	static const ap_uint<2> threshold_table[512] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0};
// Threshold Table is generated as follows:
/*
 * 	for( i = -255; i <= 255; i++ )
 * 		threshold_table[i+255] = (i < -THRESHOLD ? 1 : i > THRESHOLD ? 2 : 0);
*/
	get_image(in, img);
	eval_FAST(out, img, threshold_table);
}
