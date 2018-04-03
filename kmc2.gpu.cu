#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <list>
#include <vector>
#include <sys/time.h>
#include <stdint.h>
#include <omp.h>
#include <cuda.h>

#include "mmer.h"
#include "kmer.h"
#include "kxmer_set.h"

#define DEVE
#define STATS_FASTQ_SIZE (1 << 27)
#define STATS_READ_NUM 1278085
#define READS_PER_LOAD (1 << 21)
#define READ_LENGTH 100
#define NUM_BYTES_PER_READ 32
#define NUM_BLOCKS 2497
#define NUM_THREADS_PER_BLOCK 128
#define SIGNATURE_LEN 7
#define KMER_LENGTH 50
#define MAP_SIZE ((1 << 2 * SIGNATURE_LEN) + 1)
#define CODES_SIZE 256
#define BIN_PART_SIZE (1 << 16)
#define NUM_PARTS 38
#define BIN_NO 512
#define TEMP_FILE_BUFFER (24 * (1 << 20))
#define POS_ARRAY_SIZE 1024
#define MAX_K 256
#define KMER_WORDS 2
#define KMER_X 3
#define ALIGNMENT 0x100
#define EXPAND_BUFFER_RECS (1 << 16)
#define CUTOFF_MIN 2
#define CUTOFF_MAX 1000000000
#define NUM_STREAM 4	

#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define NORM(x, lower, upper)	((x) < (lower) ? (lower) : (x) > (upper) ? (upper) : (x))
#define BYTE_LOG(x) (((x) < (1 << 8)) ? 1 : ((x) < (1 << 16)) ? 2 : ((x) < (1 << 24)) ? 3 : 4)
#define USE_NVTX

#ifdef USE_NVTX
#include "nvToolsExt.h"
const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

using namespace std;

__constant__ char code[CODES_SIZE];

struct Cmp {
__host__ __device__  bool operator()(const CKmer &x, const CKmer &y) {
	for(int i = 1; i >= 0; --i)
		if(x.data[i] < y.data[i])
			return true;
		else if(x.data[i] > y.data[i])
			return false;
	return false;
	}
};


class binDesc
{
	public:
	int64_t bin_desc_size[BIN_NO];
	uint64_t bin_desc_n_rec[BIN_NO];
	uint64_t bin_desc_n_plus_x_recs[BIN_NO];
	uint64_t bin_desc_n_super_kmers[BIN_NO];
};

class completerDesc
{
	public: 
	unsigned char* data[BIN_NO];
	uint64_t data_size[BIN_NO];
	unsigned char* lut[BIN_NO];
	uint64_t lut_size[BIN_NO];
	uint64_t _n_unique[BIN_NO];
	uint64_t _n_cutoff_min[BIN_NO];
	uint64_t _n_cutoff_max[BIN_NO];
	uint64_t _n_total[BIN_NO];
};

class Comp
{
	uint32_t* signature_occurences;
	public:
		__host__ Comp(uint32_t* _signature_occurences) : signature_occurences(_signature_occurences){}
		__host__ bool operator()(int i, int j)
		{
			return signature_occurences[i] > signature_occurences[j];
		}
};


__host__ void GetNextSymb(unsigned char& symb, unsigned char& byte_shift, uint64_t& pos, unsigned char* data_p)
{
	symb = (data_p[pos] >> byte_shift) & 3;
//	printf("line 223 reached\n");
	if (byte_shift == 0)
	{
		pos++;
		byte_shift = 6;
	}
	else
		byte_shift -= 2;
//	printf("pos is: %d\n", pos);
}


__host__ void FromChildThread(CKmer *input_array , CKmer* thread_buffer, uint64_t size, uint32_t& input_pos)
{
	memcpy(input_array + input_pos, thread_buffer, size * (sizeof(CKmer)));
	//for(int i = 0; i < size; i++)
	//	H.push_back(thread_buffer[i]);
	input_pos += (uint32_t)size;
}

__host__ void expandKxmers(unsigned char *file_buffer, CKmer *input_array, uint64_t start_pos, uint64_t end_pos, uint32_t max_x, unsigned char* lut, uint32_t& input_pos) 
{
	//	printf("function being called\n");
		unsigned char _raw_buffer[1 << 21];
		CKmer* buffer = (CKmer*)_raw_buffer;

		CKmer kmer, rev_kmer, kmer_mask;
		CKmer  kxmer_mask;
		bool kmer_lower; //true if kmer is lower than its rev. comp
		uint32_t x, additional_symbols;
		unsigned char symb;
		uint32_t kmer_bytes = (KMER_LENGTH + 3) / 4;
		uint32_t rev_shift = KMER_LENGTH * 2 - 2;
		unsigned char *data_p = file_buffer;
		kmer_mask.set_n_1(KMER_LENGTH * 2);
		uint32_t kmer_shr = KMER_WORDS * 32 - KMER_LENGTH;

		kxmer_mask.set_n_1((KMER_LENGTH + max_x + 1) * 2);

		uint64_t buffer_pos = 0;
		uint64_t pos = start_pos;	

	//	printf("start is: %u, end is: %u \n", start_pos, end_pos);

		while (pos < end_pos)
		{
			kmer.clear();
			rev_kmer.clear();

			additional_symbols = data_p[pos++];

			//building kmer
			for (uint32_t i = 0, kmer_pos = 8 * KMER_WORDS - 1, kmer_rev_pos = 0; (int)i < (int)kmer_bytes; ++i, --kmer_pos, ++kmer_rev_pos)
			{
				kmer.set_byte(kmer_pos, data_p[pos + i]);
				rev_kmer.set_byte(kmer_rev_pos, lut[data_p[pos + i]]);
			}
			pos += (uint64_t)kmer_bytes;
			unsigned char byte_shift = 6 - (KMER_LENGTH % 4) * 2;
			if (byte_shift != 6)
				--pos;

			if (kmer_shr)
				kmer.SHR(kmer_shr);

			kmer.mask(kmer_mask);
			rev_kmer.mask(kmer_mask);

			kmer_lower = kmer < rev_kmer;
			x = 0;
			if (kmer_lower)
				buffer[buffer_pos].set(kmer);
			else
				buffer[buffer_pos].set(rev_kmer);

		//	printf("buffer_pos is: %d; pos is: %d; end_pos is: %d \n", buffer_pos, pos, end_pos);


			uint32_t symbols_left = additional_symbols;
			
			//printf("symbols_left is: %u\n", symbols_left);
			while (symbols_left != 0)
			{
		//		printf("line 299 reached\n");
				GetNextSymb(symb, byte_shift, pos, data_p);
				kmer.SHL_insert_2bits(symb);
				kmer.mask(kmer_mask);
				rev_kmer.SHR_insert_2bits(3 - symb, rev_shift);
				--symbols_left;

		//		printf("line 303 reached\n");

		//		printf("309 pos is: %ld\n", pos);
				if (kmer_lower)
				{
					if (kmer < rev_kmer)
					{
						buffer[buffer_pos].SHL_insert_2bits(symb);
						++x;
						if (x == max_x)
						{
							if (symbols_left == 0)
								break;

							buffer[buffer_pos++].set_2bits(x, KMER_LENGTH * 2 + max_x * 2);
							if((int)buffer_pos >= EXPAND_BUFFER_RECS)
							{
								FromChildThread(input_array, buffer, buffer_pos, input_pos);
								buffer_pos = 0;
							}
							x = 0;

							GetNextSymb(symb, byte_shift, pos, data_p);
							kmer.SHL_insert_2bits(symb);
							kmer.mask(kmer_mask);
							rev_kmer.SHR_insert_2bits(3 - symb, rev_shift);
							--symbols_left;

							kmer_lower = kmer < rev_kmer;

							if (kmer_lower)
								buffer[buffer_pos].set(kmer);
							else
								buffer[buffer_pos].set(rev_kmer);
						}
		//				printf("line 338 reached\n");
					}
					else
					{
						buffer[buffer_pos++].set_2bits(x, KMER_LENGTH * 2 + max_x * 2);
						if ((int)buffer_pos >= EXPAND_BUFFER_RECS)
						{
							FromChildThread(input_array, buffer, buffer_pos, input_pos);
							buffer_pos = 0;
						}
						x = 0;

						kmer_lower = false;
						buffer[buffer_pos].set(rev_kmer);
		//				printf("line 352 reached\n");
					}
				}
				else
				{
					if (!(kmer < rev_kmer))
					{
						buffer[buffer_pos].set_2bits(3 - symb, KMER_LENGTH * 2 + x * 2);
						++x;
						if (x == max_x)
						{
							if (symbols_left == 0)
								break;

							buffer[buffer_pos++].set_2bits(x, KMER_LENGTH * 2 + max_x * 2);
							if ((int)buffer_pos >= EXPAND_BUFFER_RECS)
							{
								FromChildThread(input_array, buffer, buffer_pos, input_pos);
								buffer_pos = 0;
							}
							x = 0;

							GetNextSymb(symb, byte_shift, pos, data_p);
							kmer.SHL_insert_2bits(symb);
							kmer.mask(kmer_mask);
							rev_kmer.SHR_insert_2bits(3 - symb, rev_shift);
							--symbols_left;

							kmer_lower = kmer < rev_kmer;

							if (kmer_lower)
								buffer[buffer_pos].set(kmer);
							else
								buffer[buffer_pos].set(rev_kmer);
						}
		//				printf("line 387 reached\n");
					}
					else
					{
						buffer[buffer_pos++].set_2bits(x, KMER_LENGTH * 2 + max_x * 2);
						if ((int)buffer_pos >= EXPAND_BUFFER_RECS)
						{
							FromChildThread(input_array, buffer, buffer_pos, input_pos);
							buffer_pos = 0;
						}
						x = 0;

						buffer[buffer_pos].set(kmer);
						kmer_lower = true;
					}
		//			printf("line 402 reached\n");
				}
		//		printf("407 pos is: %ld\n", pos);
				
			}
		//	printf("line 406 reached\n");
			buffer[buffer_pos++].set_2bits(x, KMER_LENGTH * 2 + max_x * 2);
			if ((int)buffer_pos >= EXPAND_BUFFER_RECS)
			{
				FromChildThread(input_array, buffer, buffer_pos, input_pos);
				buffer_pos = 0;
			}
			if (byte_shift != 6)
				++pos;
		//	printf("419 pos is: %ld\n", pos);
		//	printf("line 414 reached\n");
		}
		if (buffer_pos > 0)
		{
			FromChildThread(input_array, buffer, buffer_pos, input_pos);
			buffer_pos = 0;
		}


	}
	
	__host__ void initfiles(FILE** files) {
		string f_name;
		
		for(int i = 0; i < BIN_NO; i++) {
			string s_tmp;
			stringstream convert;
			convert << i;
			s_tmp = convert.str();
			while(s_tmp.length() < 5) {
				s_tmp = string("0") + s_tmp;
			}
			f_name = "kmc_" + s_tmp + ".bin";
			files[i] = fopen(f_name.c_str(), "wb+");
			fclose(files[i]);
		}

	}

	__host__ void openfiles(FILE** files)
	{
		string f_name;
		
		for(int i = 0; i < BIN_NO; i++) {
			string s_tmp;
			stringstream convert;
			convert << i;
			s_tmp = convert.str();
			while(s_tmp.length() < 5) {
				s_tmp = string("0") + s_tmp;
			}
			f_name = "kmc_" + s_tmp + ".bin";
			files[i] = fopen(f_name.c_str(), "ab+");
		}
	};

	__host__ void deletefiles()
        {
                string f_name;

                for(int i = 0; i < BIN_NO; i++) {
                        string s_tmp;
                        stringstream convert;
                        convert << i;
                        s_tmp = convert.str();
                        while(s_tmp.length() < 5) {
                                s_tmp = string("0") + s_tmp;
                        }
                        f_name = "kmc_" + s_tmp + ".bin";
                        remove(f_name.c_str());
                }
        };


	__host__ void closefiles(FILE** files) 
	{
		for(int i = 0; i < BIN_NO; i++) {
			fclose(files[i]);
		}	
	}

	__host__ void rewindfiles(FILE** files) 
	{
		string f_name;
		
		for(int i = 0; i < BIN_NO; i++) {
			string s_tmp;
			stringstream convert;
			convert << i;
			s_tmp = convert.str();
			while(s_tmp.length() < 5) {
				s_tmp = string("0") + s_tmp;
			}
			f_name = "kmc_" + s_tmp + ".bin";
			files[i] = fopen(f_name.c_str(), "rb+");
		}

	}
	
	__host__ void init_bin_desc(binDesc &bin_desc)
	{
		for(int i = 0; i < BIN_NO; i++) {
			bin_desc.bin_desc_size[i] = 0;
			bin_desc.bin_desc_n_rec[i] = 0;
			bin_desc.bin_desc_n_plus_x_recs[i] = 0;
			bin_desc.bin_desc_n_super_kmers[i] = 0;
		}	
	}
	
	__host__ void putBinsToDisk(FILE** files, uint64_t &tmp_size, binDesc &bin_desc, unsigned char* buffer, int64_t* bin_desc_size
				    , uint64_t* bin_desc_n_rec, uint32_t* bin_desc_n_plus_x_recs, uint64_t* bin_desc_n_super_kmers, int buffer_size, int bin_size)
	{
		PUSH_RANGE("fwrite", 1)
		openfiles(files);
		//#pragma omp parallel for
		for(int stream = 0; stream < NUM_STREAM; stream++) {
			unsigned char* stream_buffer = buffer + stream * buffer_size;
			int offset = stream * BIN_NO;
			for(int i = 0; i < BIN_NO; i++) {
				if(bin_desc_size[i + offset] != 0)
					fwrite(stream_buffer + i * bin_size, 1, bin_desc_size[i + offset], files[i]);	
				tmp_size += bin_desc_size[i + offset];
				bin_desc.bin_desc_size[i] += bin_desc_size[i + offset];
				bin_desc.bin_desc_n_rec[i] += bin_desc_n_rec[i + offset];
				bin_desc.bin_desc_n_plus_x_recs[i] += bin_desc_n_plus_x_recs[i + offset];
				bin_desc.bin_desc_n_super_kmers[i] += bin_desc_n_super_kmers[i + offset];
			}
		}
		closefiles(files);
		POP_RANGE
	}
	
	__host__ void calc_codes(char *codes) {
		for(int i = 0; i < 256; i++)
			codes[i] = -1;
		codes['A'] = codes['a'] = 0;
		codes['C'] = codes['c'] = 1;
		codes['G'] = codes['g'] = 2;
		codes['T'] = codes['t'] = 3;
	}

	__host__ uint64_t round_up_to_alignment(int64_t x)
	{
		return (x + ALIGNMENT-1) / ALIGNMENT * ALIGNMENT;
	}




	__host__ void init(uint32_t* stats, int * signature_map) {

		int n_bins = BIN_NO;
		uint32_t sorted[MAP_SIZE];
		for(uint32_t i = 0; i < MAP_SIZE; i++)
		{
			sorted[i] = i;
		}
		sort(sorted, sorted + MAP_SIZE, Comp(stats));

	/*	FILE* out;
		out = fopen("sorted.txt","w");
		
		for(int i = 0; i < (1 << SIGNATURE_LEN * 2) + 1; i++) {
			fprintf(out, "%d: %d\n", i, sorted[i]);
		}
		fclose(out);	
	*/

		list<uint32_t> _stats0;
		list<uint64_t> _stats1;	

		for (uint32_t i = 0; i < MAP_SIZE ; ++i)
		{
			if (CMmer::is_allowed(sorted[i], SIGNATURE_LEN))
			{
				_stats0.push_back(sorted[i]);
				_stats1.push_back(stats[sorted[i]]);
			}
		}

		list<uint32_t> group0;
		list<uint64_t> group1;
	//	uint32_t groupStartPos = 0;
	//	uint32_t groupEndPos = 0;	

		int bin_no = 0;
		//counting sum
		double sum = 0.0;
		for (list<uint64_t>::iterator i = _stats1.begin(); i != _stats1.end(); i++)
		{
			*i += (uint64_t)1000;
			sum += (double)*i;
		}

		double mean = sum / n_bins;
		double max_bin_size = 1.1 * mean;
		uint32_t n = n_bins - 1; //one is needed for disabled signatures
		uint32_t max_bins = n_bins - 1;

		while (_stats1.size() > n)
		{
			uint32_t max0 = _stats0.front();
			uint64_t max1 = _stats1.front();

			if (max1 > mean)
			{
				signature_map[max0] = bin_no++;				
				sum -= max1;
				mean = sum / (max_bins - bin_no);
				max_bin_size = 1.1 * mean;

				_stats0.pop_front();
				_stats1.pop_front();
				--n;
			}
			else
			{
				//heuristic
				group0.clear();
				group1.clear();
				double tmp_sum = 0.0;
				uint32_t in_current = 0;
				list<uint64_t>::iterator it1 = _stats1.begin();
				for (list<uint32_t>::iterator it0 = _stats0.begin(); it0 != _stats0.end();)
				{
					if (tmp_sum + *it1 < max_bin_size)
					{
						tmp_sum += *it1;
						group0.push_back(*it0);
						group1.push_back(*it1);
						it0 = _stats0.erase(it0);
						it1 = _stats1.erase(it1);
						++in_current;
					}
					else
					{	
						++it0;
						++it1;
					}
				}

				list<uint32_t>::iterator i0 = group0.begin();
				list<uint64_t>::iterator i1 = group1.begin();
				for (; i0 != group0.end(); ++i0)
				{
					signature_map[*i0] = bin_no;
				}
				--n;
				++bin_no;

				sum -= tmp_sum;
				mean = sum / (max_bins - bin_no);
				max_bin_size = 1.1 * mean;
			}
		}

		if (_stats0.size() > 0)
		{
			for (list<uint32_t>::iterator i = _stats0.begin(); i != _stats0.end(); ++i)
			{
				signature_map[*i] = bin_no++;
				//cout << "rest bin: " << i->second << "\n";
			}
		}
		signature_map[1 << 2 * SIGNATURE_LEN] = bin_no;



	}


	__global__ void calcStats(char* readList, uint32_t* statsBin, uint32_t* norm7)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if(tid >= STATS_READ_NUM)
			return;
		char* read = readList + tid * NUM_BYTES_PER_READ * 4;
		
		char seq[READ_LENGTH];
		uint32_t signatureStartPos;

		CMmer currentSignature, endMmer;

		char c;
		for(int i = 0; i < READ_LENGTH; i++) {
			c = read[i];
			if(c < 32)
				break;
			seq[i] = code[c];
		}	

		uint32_t i;
		uint32_t len;//length of extended kmer
		

		i = 0;
		len = 0;
		while (i + KMER_LENGTH - 1 < READ_LENGTH)
		{	
			bool contains_N = false;
			//building first signature after 'N' or at the read begining
			for (uint32_t j = 0; j < SIGNATURE_LEN; ++j, ++i)
			if (seq[i] < 0)//'N'
			{
				contains_N = true;
				break;
			}
			//signature must be shorter than k-mer so if signature contains 'N', k-mer will contains it also
			if (contains_N)
			{
				++i;
				continue;
			}
			len = SIGNATURE_LEN;
			signatureStartPos = i - SIGNATURE_LEN;
			currentSignature.insert(seq + signatureStartPos, norm7);
			endMmer.set(currentSignature);
			for (; i < READ_LENGTH; ++i)
			{
				if (seq[i] < 0)//'N'
				{
					if (len >= KMER_LENGTH)
						atomicAdd(&(statsBin[currentSignature.get()]), 1 + len - KMER_LENGTH);
					len = 0;
					++i;
					break;
				}
				endMmer.insert(seq[i], norm7);
				if (endMmer < currentSignature)//signature at the end of current k-mer is lower than current
				{
					if (len >= KMER_LENGTH)
					{
						atomicAdd(&(statsBin[currentSignature.get()]), 1 + len - KMER_LENGTH);
						len = KMER_LENGTH - 1;
					}
					currentSignature.set(endMmer);
					signatureStartPos = i - SIGNATURE_LEN + 1;
				}
				else if (endMmer == currentSignature)
				{
					currentSignature.set(endMmer);
					signatureStartPos = i - SIGNATURE_LEN + 1;
				}
				else if (signatureStartPos + KMER_LENGTH - 1 < i)//need to find new signature
				{
					atomicAdd(&(statsBin[currentSignature.get()]), 1 + len - KMER_LENGTH);
					len = KMER_LENGTH - 1;
					//looking for new signature
					++signatureStartPos;
					//building first signature in current k-mer
					endMmer.insert(seq + signatureStartPos, norm7);
					currentSignature.set(endMmer);
					for (uint32_t j = signatureStartPos + SIGNATURE_LEN; j <= i; ++j)
					{
						endMmer.insert(seq[j], norm7);
						if (endMmer <= currentSignature)
						{
							currentSignature.set(endMmer);
							signatureStartPos = j - SIGNATURE_LEN + 1;
						}
					}
				}
				++len;
			}
		}
		if (len >= KMER_LENGTH)//last one in read
			atomicAdd(&(statsBin[currentSignature.get()]), 1 + len - KMER_LENGTH);

		return;
	}

	__global__ void copy_key(uint64_t* d_key, CKmer* d_data, int idx) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		d_key[tid] = d_data[tid].data[idx];
	}

	__device__ bool is_allowed(uint32_t mmer, uint32_t len)
	{
		if ((mmer & 0x3f) == 0x3f)            // TTT suffix
			return false;
		if ((mmer & 0x3f) == 0x3b)            // TGT suffix
			return false;
		if ((mmer & 0x3c) == 0x3c)            // TG* suffix
			return false;

		for (uint32_t j = 0; j < len - 3; ++j)
		if ((mmer & 0xf) == 0)                // AA inside
			return false;
		else
			mmer >>= 2;

		if (mmer == 0)            // AAA prefix
			return false;
		if (mmer == 0x04)        // ACA prefix
			return false;
		if ((mmer & 0xf) == 0)    // *AA prefix
			return false;
	
		return true;
	}

	__device__ uint32_t get_rev(uint32_t mmer, uint32_t len)
	{
		uint32_t rev = 0;
		uint32_t shift = len*2 - 2;
		for(uint32_t i = 0 ; i < len ; ++i)
		{
			rev += (3 - (mmer & 3)) << shift;
			mmer >>= 2;
			shift -= 2;
		}
		return rev;
	}



	__global__ void calc_norm(uint32_t* norm7) {
		uint32_t special = 1 << 7 * 2;
		uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	        if(i < special)
	  	{
		                uint32_t rev = get_rev(i, 7);
		                uint32_t str_val = is_allowed(i, 7) ? i : special;
		                uint32_t rev_val = is_allowed(rev, 7) ? rev : special;
		                norm7[i] = MIN(str_val, rev_val);
		}
	}

	__device__ void update_n_plus_x_recs(char* seq, uint32_t n, uint32_t &bin_desc_n_plus_x_recs) {
		unsigned char kmer, rev;
		uint32_t kmer_pos = 4;
		uint32_t rev_pos = KMER_LENGTH;
		uint32_t x;

		uint32_t n_plus_x_recs_tmp = 0;

		kmer = (seq[0] << 6) + (seq[1] << 4) + (seq[2] << 2) + seq[3];
		rev = ((3 - seq[KMER_LENGTH - 1]) << 6) + ((3 - seq[KMER_LENGTH - 2]) << 4) + ((3 - seq[KMER_LENGTH - 3]) << 2) + (3 - seq[KMER_LENGTH - 4]);

		x = 0;
		//kmer_smaller: 0, rev_smaller: 1, equals: 2
		int current_state, new_state;
		if (kmer < rev)
			current_state = 0;
		else if (rev < kmer)
			current_state = 1;
		else
			current_state = 2;


		for (uint32_t i = 0; i < n - KMER_LENGTH; ++i)
		{
			rev >>= 2;
			rev += (3 - seq[rev_pos++]) << 6;
			kmer <<= 2;
			kmer += seq[kmer_pos++];

			if (kmer < rev)
				new_state = 0;
			else if (rev < kmer)
				new_state = 1;
			else
				new_state = 2;

			if (new_state == current_state)
			{
				if (current_state == 2)
					++n_plus_x_recs_tmp;
				else
					++x;
			}
			else
			{
				current_state = new_state;
				n_plus_x_recs_tmp += 1 + x / 4;

				x = 0;
			}
		}
		n_plus_x_recs_tmp += 1 + x / 4;

		atomicAdd(&(bin_desc_n_plus_x_recs), n_plus_x_recs_tmp);
	}
	

	__device__ void putExtendedKmer(char* seq, uint32_t n, unsigned char* buffer, int64_t &bin_desc_size, uint64_t &bin_desc_n_rec
				  , uint32_t &bin_desc_n_plus_x_recs, uint64_t &bin_desc_n_super_kmers, int &lock) {
		volatile bool leaveLoop = false;
		while(!leaveLoop) 
		{
			if(atomicCAS(&lock, 0, 1) == 0)
			{
				buffer[bin_desc_size++] = n - KMER_LENGTH;		
				for(uint32_t i = 0, j = 0 ; i < n / 4 ; ++i,j+=4)
					buffer[bin_desc_size++] = (seq[j] << 6) + (seq[j + 1] << 4) + (seq[j + 2] << 2) + seq[j + 3];
				switch (n%4)
				{
					case 1:
						buffer[bin_desc_size++] = (seq[n-1] << 6);
						break;
					case 2:
						buffer[bin_desc_size++] = (seq[n-2] << 6) + (seq[n-1] << 4);
						break;
					case 3:
						buffer[bin_desc_size++] = (seq[n-3] << 6) + (seq[n-2] << 4) + (seq[n-1] << 2);
						break;
				}

				++bin_desc_n_super_kmers;
				bin_desc_n_rec += n - KMER_LENGTH + 1;						
			
				update_n_plus_x_recs(seq, n, bin_desc_n_plus_x_recs); 
				leaveLoop = true;
				atomicExch(&lock, 0);
			}		
		}
	};

	__global__ void processReads(char* readList, int* sigMap,  uint32_t* statsBin, unsigned char* buffer, int64_t* bin_desc_size, uint64_t* bin_desc_n_rec
				  , uint32_t* bin_desc_n_plus_x_recs, uint64_t* bin_desc_n_super_kmers, int* lock, int counter, uint32_t* norm7, int offset, int bin_size)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		//__shared__ shared_bins shared;

		//shared.init_bins();
		
		if(tid >= counter)
			return;
		//atomicAdd(&(statsBin[0]), 1);
		char* read = readList + (tid + offset) * NUM_BYTES_PER_READ * 4;
		
		char seq[READ_LENGTH];
		uint32_t signatureStartPos;

		CMmer currentSignature, endMmer;

		unsigned char c;
		for(int i = 0; i < READ_LENGTH; i++) {
			c = read[i];
			if(c < 32)
				break;
			seq[i] = code[c];
		}	

		uint32_t i;
		uint32_t len;//length of extended kmer

		int idx = 0;
		uint16_t lenVec[READ_LENGTH - KMER_LENGTH + 1];
		uint16_t startPosVec[READ_LENGTH - KMER_LENGTH + 1];		
		int bin_no[READ_LENGTH - KMER_LENGTH + 1];	

		i = SIGNATURE_LEN;

		len = SIGNATURE_LEN;
		signatureStartPos = i - SIGNATURE_LEN;
		currentSignature.insert(seq + signatureStartPos, norm7);
		endMmer.set(currentSignature);
		for (; i < READ_LENGTH; ++i)
		{
			endMmer.insert(seq[i], norm7);
			if (endMmer < currentSignature)//signature at the end of current k-mer is lower than current
			{
				if (len >= KMER_LENGTH)
				{
					bin_no[idx] = sigMap[currentSignature.get()];
					lenVec[idx] = len;
					startPosVec[idx++] = i - len;
					len = KMER_LENGTH - 1;
				}
				currentSignature.set(endMmer);
				signatureStartPos = i - SIGNATURE_LEN + 1;
			}
			else if (endMmer == currentSignature)
			{
				currentSignature.set(endMmer);
				signatureStartPos = i - SIGNATURE_LEN + 1;
			}
			else if (signatureStartPos + KMER_LENGTH - 1 < i)//need to find new signature
			{
				bin_no[idx] = sigMap[currentSignature.get()];
				lenVec[idx] = len;
				startPosVec[idx++] = i - len;

				len = KMER_LENGTH - 1;
				//looking for new signature
				++signatureStartPos;
				//building first signature in current k-mer
				endMmer.insert(seq + signatureStartPos, norm7);
				currentSignature.set(endMmer);
				for (uint32_t j = signatureStartPos + SIGNATURE_LEN; j <= i; ++j)
				{
					endMmer.insert(seq[j], norm7);
					if (endMmer <= currentSignature)
					{
						currentSignature.set(endMmer);
						signatureStartPos = j - SIGNATURE_LEN + 1;
					}
				}
			}
			++len;
		}
		
		if (len >= KMER_LENGTH)//last one in read
		{
			bin_no[idx] = sigMap[currentSignature.get()];	
			lenVec[idx] = len;
			startPosVec[idx++] = i - len;
		}	
	
		for(i = 0; i < idx; i++) {
			int no = bin_no[i];
			putExtendedKmer(seq + startPosVec[i], lenVec[i], buffer+bin_size * no, bin_desc_size[no]
					,bin_desc_n_rec[no],bin_desc_n_plus_x_recs[no],bin_desc_n_super_kmers[no], lock[no]);
		}
		/*
int tid = threadIdx.x + blockIdx.x * blockDim.x;

		//__shared__ shared_bins shared;

		//shared.init_bins();
		
		if(tid >= counter)
			return;
		//atomicAdd(&(statsBin[0]), 1);
		char* read = readList + tid * NUM_BYTES_PER_READ * 4;
		
		char seq[READ_LENGTH];
		uint32_t signatureStartPos;

		CMmer currentSignature, endMmer;
		int bin_no;	

		unsigned char c;
		for(int i = 0; i < READ_LENGTH; i++) {
			c = read[i];
			if(c < 32)
				break;
			seq[i] = code[c];
		}	

		uint32_t i;
		uint32_t len;//length of extended kmer

		i = 0;
		len = 0;
		while (i + KMER_LENGTH - 1 < READ_LENGTH)
		{	
			bool contains_N = false;
			//building first signature after 'N' or at the read begining
			for (uint32_t j = 0; j < SIGNATURE_LEN; ++j, ++i)
			if (seq[i] < 0)//'N'
			{
				contains_N = true;
				break;
			}
			//signature must be shorter than k-mer so if signature contains 'N', k-mer will contains it also
			if (contains_N)
			{
				++i;
				continue;
			}
			len = SIGNATURE_LEN;
			signatureStartPos = i - SIGNATURE_LEN;
			currentSignature.insert(seq + signatureStartPos, norm7);
			endMmer.set(currentSignature);
			for (; i < READ_LENGTH; ++i)
			{
				if (seq[i] < 0)//'N'
				{
					if (len >= KMER_LENGTH)
					{
						bin_no = sigMap[currentSignature.get()];	
						putExtendedKmer(seq + i - len, len, buffer+BIN_PART_SIZE * NUM_PARTS * bin_no, bin_desc_size[bin_no]
								,bin_desc_n_rec[bin_no],bin_desc_n_plus_x_recs[bin_no],bin_desc_n_super_kmers[bin_no], lock[bin_no]);
					}
					len = 0;
					++i;
					break;
				}
				endMmer.insert(seq[i], norm7);
				if (endMmer < currentSignature)//signature at the end of current k-mer is lower than current
				{
					if (len >= KMER_LENGTH)
					{
						bin_no = sigMap[currentSignature.get()];
						putExtendedKmer(seq + i - len, len, buffer+BIN_PART_SIZE * NUM_PARTS * bin_no, bin_desc_size[bin_no]
								,bin_desc_n_rec[bin_no],bin_desc_n_plus_x_recs[bin_no],bin_desc_n_super_kmers[bin_no], lock[bin_no]);

						len = KMER_LENGTH - 1;
					}
					currentSignature.set(endMmer);
					signatureStartPos = i - SIGNATURE_LEN + 1;
				}
				else if (endMmer == currentSignature)
				{
					currentSignature.set(endMmer);
					signatureStartPos = i - SIGNATURE_LEN + 1;
				}
				else if (signatureStartPos + KMER_LENGTH - 1 < i)//need to find new signature
				{
					bin_no = sigMap[currentSignature.get()];
					putExtendedKmer(seq + i - len, len, buffer+BIN_PART_SIZE * NUM_PARTS * bin_no, bin_desc_size[bin_no]
							,bin_desc_n_rec[bin_no],bin_desc_n_plus_x_recs[bin_no],bin_desc_n_super_kmers[bin_no], lock[bin_no]);

					len = KMER_LENGTH - 1;
					//looking for new signature
					++signatureStartPos;
					//building first signature in current k-mer
					endMmer.insert(seq + signatureStartPos, norm7);
					currentSignature.set(endMmer);
					for (uint32_t j = signatureStartPos + SIGNATURE_LEN; j <= i; ++j)
					{
						endMmer.insert(seq[j], norm7);
						if (endMmer <= currentSignature)
						{
							currentSignature.set(endMmer);
							signatureStartPos = j - SIGNATURE_LEN + 1;
						}
					}
				}
				++len;
				if(len == KMER_LENGTH + 255)
				{
					bin_no = sigMap[currentSignature.get()];
					putExtendedKmer(seq + i + 1 - len, len, buffer+BIN_PART_SIZE * NUM_PARTS * bin_no, bin_desc_size[bin_no]
							,bin_desc_n_rec[bin_no],bin_desc_n_plus_x_recs[bin_no],bin_desc_n_super_kmers[bin_no], lock[bin_no]);

					i -= KMER_LENGTH - 2;
					len = 0;
					break;
				}
			}
		}
		if (len >= KMER_LENGTH)//last one in read
		{
			bin_no = sigMap[currentSignature.get()];	
			putExtendedKmer(seq + i - len, len, buffer+BIN_PART_SIZE * NUM_PARTS * bin_no, bin_desc_size[bin_no]
					,bin_desc_n_rec[bin_no],bin_desc_n_plus_x_recs[bin_no],bin_desc_n_super_kmers[bin_no], lock[bin_no]);

		}	


	*/
		return;
	}

	__global__ void init_bins(unsigned char* buffer, int64_t* bin_desc_size, uint64_t* bin_desc_n_rec
				  , uint32_t* bin_desc_n_plus_x_recs, uint64_t* bin_desc_n_super_kmers, int* lock)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if(tid > 511)
			return;

		lock[tid] = 0;
		
		bin_desc_size[tid] = 0;
		bin_desc_n_rec[tid] = 0;
		bin_desc_n_plus_x_recs[tid] = 0;
		bin_desc_n_super_kmers[tid] = 0;
		
		return;
	}


	__host__ void PreCompactKxmers(CKmer* &sort_buffer, uint64_t& compacted_count, uint32_t* &kxmer_counters, uint64_t n_plus_x_recs)
	{
		compacted_count = 0;

		CKmer *act_kmer;
		act_kmer = &sort_buffer[0];
		kxmer_counters[compacted_count] = 1;

		for (uint32_t i = 1; i < n_plus_x_recs; ++i)
		{
			if (*act_kmer == sort_buffer[i])
				++kxmer_counters[compacted_count];
			else
			{
				sort_buffer[compacted_count++] = *act_kmer;
				kxmer_counters[compacted_count] = 1;
				act_kmer = &sort_buffer[i];
			}
		}
		sort_buffer[compacted_count++] = *act_kmer;

	}

	__host__ uint64_t FindFirstSymbOccur(CKmer* sort_buffer, uint64_t start_pos, uint64_t end_pos, uint32_t offset, unsigned char symb, uint32_t max_x)
	{
		uint32_t kxmer_offset = (KMER_LENGTH + max_x - offset) * 2;
		uint64_t middle_pos;
		unsigned char middle_symb;
		while (start_pos < end_pos)
		{
			middle_pos = (start_pos + end_pos) / 2;
			middle_symb = sort_buffer[middle_pos].get_2bits(kxmer_offset);
			if (middle_symb < symb)
				start_pos = middle_pos + 1;
			else
				end_pos = middle_pos;
		}
		return end_pos;
	}

	__host__ void InitKXMerSet(CKXmerSet& kxmer_set, CKmer* sort_buffer, uint64_t start_pos, uint64_t end_pos, uint32_t offset, uint32_t depth, uint32_t max_x)
	{
		if (start_pos == end_pos)
			return;
		uint32_t shr = max_x + 1 - offset;
		kxmer_set.init_add(start_pos, end_pos, shr);

		--depth;
		if (depth > 0)
		{
			uint64_t pos[5];
			pos[0] = start_pos;
			pos[4] = end_pos;
			for (uint32_t i = 1; i < 4; ++i)
				pos[i] = FindFirstSymbOccur(sort_buffer, pos[i - 1], end_pos, offset, i, max_x);
			for (uint32_t i = 1; i < 5; ++i)
				InitKXMerSet(kxmer_set, sort_buffer, pos[i - 1], pos[i], offset + 1, depth, max_x);
		}
	}




	__host__ void compact_kxmers(CKmer* sort_buffer, uint32_t lut_prefix_len, unsigned char* buffer, uint64_t part1_size, uint64_t out_buffer_size, 
				     uint64_t in_n_plus_x_recs, uint64_t kxmers_size, uint32_t max_x, uint64_t counter_max, completerDesc &completer, int bin_no)
	{

		CKXmerSet kxmer_set(KMER_LENGTH);
		kxmer_set.clear();
		kxmer_set.set_buffer(sort_buffer);
		uint64_t n_unique = 0;
		uint64_t n_cutoff_min = 0;
		uint64_t n_cutoff_max = 0;
		uint64_t n_total = 0;

		uint32_t kmer_symbols = KMER_LENGTH - lut_prefix_len;
		uint64_t kmer_bytes = kmer_symbols / 4;
		uint64_t lut_recs = 1 << (2 * lut_prefix_len);
		uint64_t lut_size = lut_recs * sizeof(uint64_t);


		unsigned char *out_buffer = NULL;
		unsigned char *raw_lut = NULL;
		
		out_buffer = buffer + part1_size;
		raw_lut = out_buffer + out_buffer_size;

		uint64_t *lut = (uint64_t*)raw_lut;
		for(int i = 0; i < lut_recs; i++) {
			*(lut + i) = 0;
		}

		uint32_t out_pos = 0;
		uint32_t *kxmer_counters;		

		if (in_n_plus_x_recs)
		{
			unsigned char* raw_kxmer_counters = NULL;
			raw_kxmer_counters = buffer + kxmers_size;
			kxmer_counters = (uint32_t*)raw_kxmer_counters;
			uint64_t compacted_count;
			PreCompactKxmers(sort_buffer, compacted_count, kxmer_counters, in_n_plus_x_recs);

			uint64_t pos[5];//pos[symb] is first position where symb occur (at first position of k+x-mer) and pos[symb+1] jest first position where symb is not starting symbol of k+x-mer
			pos[0] = 0;
			pos[4] = compacted_count;
			for (uint32_t i = 1; i < 4; ++i)
				pos[i] = FindFirstSymbOccur(sort_buffer, pos[i - 1], compacted_count, 0, i, max_x);
			for (uint32_t i = 1; i < 5; ++i)
				InitKXMerSet(kxmer_set, sort_buffer, pos[i - 1], pos[i], max_x + 2 - i, i, max_x);

		
		

			uint64_t counter_pos = 0;
	
			uint64_t counter_size = min(BYTE_LOG(CUTOFF_MAX), BYTE_LOG(counter_max));

			CKmer kmer, next_kmer;
			CKmer kmer_mask;

			kmer.clear();
			next_kmer.clear();
	
			kmer_mask.set_n_1(KMER_LENGTH * 2);
			uint32_t count;
			//first
			kxmer_set.get_min(counter_pos, kmer);
			count = kxmer_counters[counter_pos];
			//rest
			while (kxmer_set.get_min(counter_pos, next_kmer))
			{
				if (kmer == next_kmer)
					count += kxmer_counters[counter_pos];
				else
				{
					n_total += count;
					++n_unique;
					if (count < (uint32_t)CUTOFF_MIN)
						n_cutoff_min++;
					else if (count >(uint32_t)CUTOFF_MAX)
						n_cutoff_max++;
					else
					{
						lut[kmer.remove_suffix(2 * kmer_symbols)]++;
						if (count > (uint32_t)counter_max)
							count = counter_max;
	
						// Store compacted kmer

						for (int32_t j = (int32_t)kmer_bytes - 1; j >= 0; --j)
							out_buffer[out_pos++] = kmer.get_byte(j);
						for (int32_t j = 0; j < (int32_t)counter_size; ++j)
							out_buffer[out_pos++] = (count >> (j * 8)) & 0xFF;
					}
					count = kxmer_counters[counter_pos];
					kmer = next_kmer;
				}
			}

			//last one
			++n_unique;
			n_total += count;
			if (count < (uint32_t)CUTOFF_MIN)
				n_cutoff_min++;
			else if (count >(uint32_t)CUTOFF_MAX)
				n_cutoff_max++;
			else
			{
				lut[kmer.remove_suffix(2 * kmer_symbols)]++;
				if (count > (uint32_t)counter_max)
					count = counter_max;
	
				// Store compacted kmer
				for (int32_t j = (int32_t)kmer_bytes - 1; j >= 0; --j)
					out_buffer[out_pos++] = kmer.get_byte(j);
				for (int32_t j = 0; j < (int32_t)counter_size; ++j)
					out_buffer[out_pos++] = (count >> (j * 8)) & 0xFF;
			}

		}
		completer.data[bin_no] = out_buffer;
		completer.data_size[bin_no] = out_pos;
		completer.lut[bin_no] = raw_lut;
		completer.lut_size[bin_no] = lut_size;
		completer._n_unique[bin_no] = n_unique;
		completer._n_cutoff_min[bin_no] = n_cutoff_min;
		completer._n_cutoff_max[bin_no] = n_cutoff_max;
		completer._n_total[bin_no] = n_total;
	}

	__host__ bool store_uint(FILE *out, uint64_t x, uint32_t size)
	{
		for(uint32_t i = 0; i < size; ++i)
			putc((x >> (i * 8)) & 0xFF, out);

		return true;
	}


	int main(int argc, char** argv) {
		//cuda variables
		std::ios_base::sync_with_stdio(false);
		cudaError_t cerr;
		const char* cudaErrorStr;
		
		//fastq file
		FILE* fastqFile;
		
		//output file
		int reads_per_load = READS_PER_LOAD /** KMER_LENGTH / 44*/;
		/*if(KMER_LENGTH > 44)
			reads_per_load = READS_PER_LOAD;
		*/
		
		//host variables
		char* read                     = (char*) malloc(sizeof(char) * (READ_LENGTH + 2));
		char* dumm                     = (char*) malloc(sizeof(char) * (READ_LENGTH * 2 + 1));
		char* statsReadListHost        = (char*) malloc(sizeof(char) * STATS_READ_NUM * NUM_BYTES_PER_READ * 4);
		char* readListHost	       = (char*) malloc(sizeof(char) * reads_per_load * NUM_BYTES_PER_READ * 4);
		uint32_t* binStatsHost	       = (uint32_t*) malloc(sizeof(uint32_t) * ((1 << SIGNATURE_LEN * 2) + 1));
		int signature_map[MAP_SIZE];
		char* codesHost		       = (char*) malloc(sizeof(char) * CODES_SIZE);	
		binDesc binDescHost;
		FILE* binFiles[BIN_NO];
		
		unsigned char* buffer_host;//	= (unsigned char*)malloc(sizeof(unsigned char)*(BIN_PART_SIZE * NUM_PARTS * BIN_NO));
		int64_t* bin_desc_size_host;//	= (int64_t*)malloc(sizeof(int64_t)*BIN_NO*NUM_STREAM);
		uint64_t* bin_desc_n_rec_host;//	= (uint64_t*)malloc(sizeof(uint64_t)*BIN_NO*NUM_STREAM);
		uint32_t* bin_desc_n_plus_x_recs_host;// = (uint32_t*)malloc(sizeof(uint32_t)*BIN_NO*NUM_STREAM);
		uint64_t* bin_desc_n_super_kmers_host;// = (uint64_t*)malloc(sizeof(uint64_t)*BIN_NO*NUM_STREAM);

		//stage two variables

		//device variables
		char* statsReadDevice;	
		char* readListDevice;
		uint32_t* binStatsDevice;
		int* sigMapDevice;
		uint32_t* norm7;
		unsigned char* buffer_device;
		int64_t* bin_desc_size_device;
		uint64_t* bin_desc_n_rec_device;
		uint32_t* bin_desc_n_plus_x_recs_device;
		uint64_t* bin_desc_n_super_kmers_device;
		int* lock_device;

		int fileAlive;
		int readCounter;

		cudaHostAlloc((void**)&buffer_host, (unsigned long)(sizeof(unsigned char)*(BIN_PART_SIZE * NUM_PARTS * BIN_NO)), 0);
		cudaHostAlloc((void**)&bin_desc_size_host, (unsigned long)(sizeof(int64_t)*(NUM_STREAM * BIN_NO)), 0);
		cudaHostAlloc((void**)&bin_desc_n_rec_host, (unsigned long)(sizeof(uint64_t)*(NUM_STREAM * BIN_NO)), 0);
		cudaHostAlloc((void**)&bin_desc_n_plus_x_recs_host, (unsigned long)(sizeof(uint32_t)*(NUM_STREAM * BIN_NO)), 0);
		cudaHostAlloc((void**)&bin_desc_n_super_kmers_host, (unsigned long)(sizeof(uint64_t)*(NUM_STREAM * BIN_NO)), 0);


		double processReadsTimer = 0;	

		timeval timeStart, timeEnd;
		gettimeofday(&(timeStart), NULL);
		deletefiles();
		cout<<"kmer length is:"<<KMER_LENGTH<<endl;
		for(int i = 0; i < MAP_SIZE; i++)
			signature_map[i] = -1;

		size_t mem_tot, mem_free;
		
		fileAlive = 1;
		readCounter = 0;
		fastqFile = fopen(argv[1], "r");
		//outFile = fopen(argv[2], "w");
		//read fastq reads into stats read buffer
		while(fileAlive > 0)
		{
			if (fileAlive > 0){ 
				if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0){ 
				if(fgets(read, READ_LENGTH + 2, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0){ 
				if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0){ 
				if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0)
			{
				//fprintf(outFile, "%s", read);
				memcpy(statsReadListHost + NUM_BYTES_PER_READ * 4 * readCounter, read, READ_LENGTH);
				readCounter++;

				if(readCounter >= STATS_READ_NUM)
					break;
			}
		}
		fclose(fastqFile);
		//set device to device 0	
		cerr = cudaSetDevice(0);
		if(cerr != cudaSuccess) {
			printf("cannot set to device 0, %s\n", cerr);
			return 1;
		}
		//prepare codes for constant memory
		calc_codes(codesHost);	

		cerr = cudaMemcpyToSymbol(code, codesHost, CODES_SIZE * sizeof(char));
		if(cerr != cudaSuccess) {
			printf("cannot copy codes to symbol\n");
			return 1;
		}
		//allocate space for stats fastq reads
		cerr = cudaMalloc(&statsReadDevice, sizeof(char) * STATS_READ_NUM * NUM_BYTES_PER_READ * 4);
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for stats reads on device\n");
			return 1;
		}

		cerr = cudaMalloc((void**)&readListDevice, sizeof(char) * reads_per_load * NUM_BYTES_PER_READ * 4);
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for reads list on device\n");
			return 1;
		}

		//allocate space for signature map
		cerr = cudaMalloc(&sigMapDevice, sizeof(int) * MAP_SIZE);
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for signature map on device\n");
			return 1;
		}

		cerr = cudaMalloc(&norm7, sizeof(int) * (1<<7*2));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for norm7  on device\n");
			return 1;
		}

		
		//allocate space for stats
		cerr = cudaMalloc(&binStatsDevice, sizeof(uint32_t) * ((1 << SIGNATURE_LEN * 2) + 1));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for stats bins on device\n");
			return 1;
		}
	
		cerr = cudaMalloc(&buffer_device, sizeof(char) * (BIN_PART_SIZE * NUM_PARTS * BIN_NO));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for buffer_device on device\n");
			return 1;
		}

		cerr = cudaMalloc(&bin_desc_size_device, sizeof(int64_t) * (BIN_NO*NUM_STREAM));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for bin_desc_size_device on device\n");
			return 1;
		}

		cerr = cudaMalloc(&bin_desc_n_rec_device, sizeof(uint64_t) * (BIN_NO*NUM_STREAM));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for bin_desc_n_rec on device\n");
			return 1;
		}

		cerr = cudaMalloc(&bin_desc_n_plus_x_recs_device, sizeof(uint32_t) * (BIN_NO*NUM_STREAM));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for bin_desc_n_plus_x_recs on device\n");
			return 1;
		}

		cerr = cudaMalloc(&bin_desc_n_super_kmers_device, sizeof(uint64_t) * (BIN_NO*NUM_STREAM));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for bin_desc_n_super_kmers on device\n");
			return 1;
		}

		cerr = cudaMalloc(&lock_device, sizeof(int) * (BIN_NO*NUM_STREAM));
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for lock on device\n");
			return 1;
		}


		//initialize stats bins to 0
		cerr = cudaMemset(binStatsDevice, 0, sizeof(uint32_t) * ((1 << SIGNATURE_LEN * 2) + 1));
		if(cerr != cudaSuccess) {
			printf("cannot initialize statsBins to 0\n");
			return 1;
		}
		
		
		//transfer stats fastq reads to device
		cerr = cudaMemcpy(statsReadDevice, statsReadListHost, sizeof(char) * STATS_READ_NUM * NUM_BYTES_PER_READ * 4, cudaMemcpyHostToDevice);
		if(cerr != cudaSuccess) {
			cudaErrorStr = cudaGetErrorString(cerr);
			printf("cannot copy stats read list to device: %s\n", cudaErrorStr); 
			return 1;
		}

		
		calc_norm<<<(1<<7*2)/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(norm7);


		
		//execution of the calcStats kernel
		calcStats<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(statsReadDevice, binStatsDevice, norm7);	
		printf("calcStats done\n");
		cerr = cudaMemcpy(binStatsHost, binStatsDevice, sizeof(uint32_t) * MAP_SIZE, cudaMemcpyDeviceToHost);
		if(cerr != cudaSuccess) {
			cudaErrorStr = cudaGetErrorString(cerr);
			printf("cannot copy binstats back to host: %s\n", cudaErrorStr); 
			return 1;
		}
		
		init(binStatsHost,signature_map);


		cerr = cudaMemcpy(sigMapDevice, signature_map, sizeof(int) * MAP_SIZE, cudaMemcpyHostToDevice);
		if(cerr != cudaSuccess) {
			cudaErrorStr = cudaGetErrorString(cerr);
			printf("cannot copy signature map to device: %s\n", cudaErrorStr); 
			return 1;
		}

		cerr = cudaMemcpy(signature_map,sigMapDevice, sizeof(int) * MAP_SIZE, cudaMemcpyDeviceToHost);

		//printf("16384 is: %d", signature_map[16384]);
		//preprocessing stage finished
		cudaFree(statsReadDevice);	
		//stage 1

		uint64_t n_reads = 0;

		uint64_t tmp_size = 0;

		fastqFile = fopen(argv[1], "r");

		double fastqReaderTimer = 0;

		int num_block = ceil((double)reads_per_load /(double)NUM_THREADS_PER_BLOCK);
		printf("num_block is: %d\n", num_block);
		
		printf("reads_per_load is: %d\n", reads_per_load);


		readCounter = 0;
		


		init_bin_desc(binDescHost);
		fileAlive = 1;

		int ctr = 0;

		while(fileAlive > 0)
		{
			if (fileAlive > 0){ 
				if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0){ 	
				if(fgets(read, READ_LENGTH + 2, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0){ 
				if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0){ 
				if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
					fileAlive = 0;
			}
			if (fileAlive > 0)
			{
				memcpy(readListHost + NUM_BYTES_PER_READ * 4 * readCounter, read, READ_LENGTH);
				readCounter++;
				ctr++;
				if(readCounter >= reads_per_load)
					break;
			}
		}

		n_reads += readCounter;
		int next_batch = fileAlive;
		int batch_num = 0;

		cudaStream_t stream[NUM_STREAM];
	
		for(int i = 0; i < NUM_STREAM; i++) 
			cudaStreamCreate(&stream[i]);
		
		int buffer_size_per_stream = BIN_PART_SIZE * NUM_PARTS * BIN_NO / NUM_STREAM; 
		int bin_size = buffer_size_per_stream / BIN_NO;
		printf("buffer size is %d\n", buffer_size_per_stream);
		while(next_batch > 0) {
			if(fileAlive <= 0)
				next_batch = 0;
			
			int readNum = readCounter / NUM_STREAM;
			int offset;
			offset = 0;
			if(0 == NUM_STREAM - 1)
				readNum = reads_per_load;
		
			cerr = cudaMemcpyAsync(readListDevice + offset, readListHost + offset, sizeof(char) * readNum * NUM_BYTES_PER_READ * 4, cudaMemcpyHostToDevice, stream[0]);
			if(cerr != cudaSuccess) {
				cudaErrorStr = cudaGetErrorString(cerr);
				printf("cannot copy read list to device: %s\n", cudaErrorStr); 
				printf("%d", readCounter);
				return 1;
			}
			

			num_block = ceil((double)(readCounter / NUM_STREAM) / (double)NUM_THREADS_PER_BLOCK);
			
			init_bins<<<1, BIN_NO, 0, stream[0]>>>(buffer_device, bin_desc_size_device, bin_desc_n_rec_device
								, bin_desc_n_plus_x_recs_device, bin_desc_n_super_kmers_device,lock_device);
			
			readNum = readCounter / NUM_STREAM ;
			
			processReads<<<num_block, NUM_THREADS_PER_BLOCK, 0, stream[0]>>>(readListDevice, sigMapDevice, binStatsDevice, buffer_device
						, bin_desc_size_device, bin_desc_n_rec_device, bin_desc_n_plus_x_recs_device
						, bin_desc_n_super_kmers_device, lock_device, readNum, norm7, 0, bin_size);	
			
			if(batch_num != 0)
				putBinsToDisk(binFiles, tmp_size, binDescHost, buffer_host, bin_desc_size_host, bin_desc_n_rec_host
						, bin_desc_n_plus_x_recs_host, bin_desc_n_super_kmers_host, buffer_size_per_stream, bin_size);

			int reads_per_stream = reads_per_load / (NUM_STREAM - 1);
			int newReadCounter = 0;	
	
			for(int i = 1; i < NUM_STREAM + 1; i++) {
				int new_offset, new_readNum;
				if(i != NUM_STREAM) {	
					new_offset = readNum * NUM_BYTES_PER_READ * 4 * i;
					if(i == NUM_STREAM - 1)
						readNum = reads_per_load - readNum * (NUM_STREAM - 1);
				
					cerr = cudaMemcpyAsync(readListDevice + new_offset, readListHost + new_offset, sizeof(char) * readNum * NUM_BYTES_PER_READ * 4, cudaMemcpyHostToDevice, stream[i]);
					if(cerr != cudaSuccess) {
						cudaErrorStr = cudaGetErrorString(cerr);
						printf("cannot copy read list to device: %s\n", cudaErrorStr); 
						printf("%d", readCounter);
						return 1;
					}
					
					new_offset = BIN_NO * i;
					init_bins<<<1, BIN_NO, 0, stream[i]>>>(buffer_device, bin_desc_size_device + new_offset, bin_desc_n_rec_device + new_offset
										, bin_desc_n_plus_x_recs_device + new_offset, bin_desc_n_super_kmers_device + new_offset,lock_device + new_offset);
					
					new_readNum = readCounter / NUM_STREAM ;
					if(i == NUM_STREAM - 1)
						new_readNum = readCounter - new_readNum * (NUM_STREAM - 1);
					
					processReads<<<num_block, NUM_THREADS_PER_BLOCK, 0, stream[i]>>>(readListDevice, sigMapDevice, binStatsDevice, buffer_device + i * buffer_size_per_stream
								, bin_desc_size_device + new_offset, bin_desc_n_rec_device + new_offset, bin_desc_n_plus_x_recs_device + new_offset
								, bin_desc_n_super_kmers_device + new_offset, lock_device + new_offset, new_readNum, norm7, (readCounter/NUM_STREAM) * i, bin_size);	
				}
				//PUSH_RANGE("buffer", 3)	
				cerr = cudaMemcpyAsync(buffer_host + (i - 1) * buffer_size_per_stream, buffer_device + (i - 1) * buffer_size_per_stream, sizeof(char) * (buffer_size_per_stream), cudaMemcpyDeviceToHost, stream[i - 1]);
				if(cerr != cudaSuccess) {
					cudaErrorStr = cudaGetErrorString(cerr);
					printf("cannot copy buffer to host: %s\n", cudaErrorStr); 
					printf("%d", readCounter);
					return 1;
				}
				//POP_RANGE
				if(i != NUM_STREAM) {
					int reads_limit = (i == NUM_STREAM - 1)?reads_per_load:(reads_per_stream * i);	
					PUSH_RANGE("fget", 2)
					while(fileAlive > 0)
					{
						if (fileAlive > 0){ 
							if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
								fileAlive = 0;
						}
						if (fileAlive > 0){ 
							if(fgets(read, READ_LENGTH + 2, fastqFile) == NULL)
								fileAlive = 0;
						}
						if (fileAlive > 0){ 
							if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
								fileAlive = 0;
						}
						if (fileAlive > 0){ 
							if(fgets(dumm, READ_LENGTH * 2 + 1, fastqFile) == NULL)
								fileAlive = 0;
						}
						if (fileAlive > 0)
						{
							memcpy(readListHost + NUM_BYTES_PER_READ * 4 * newReadCounter, read, READ_LENGTH);
							newReadCounter++;
							ctr++;
							if(newReadCounter >= reads_limit)
								break;
						}
					}
					POP_RANGE
				}
				cerr = cudaMemcpyAsync(bin_desc_size_host + offset, bin_desc_size_device + offset, sizeof(int64_t) * (BIN_NO), cudaMemcpyDeviceToHost, stream[i - 1]);
				if(cerr != cudaSuccess) {
					cudaErrorStr = cudaGetErrorString(cerr);
					printf("cannot copy bin_desc_size to host: %s %d\n", cudaErrorStr, offset); 
					return 1;
				}
		
				cerr = cudaMemcpyAsync(bin_desc_n_rec_host + offset, bin_desc_n_rec_device + offset, sizeof(uint64_t) * (BIN_NO), cudaMemcpyDeviceToHost, stream[i - 1]);
				if(cerr != cudaSuccess) {
					cudaErrorStr = cudaGetErrorString(cerr);
					printf("cannot copy bin_desc_n_rec to host: %s\n", cudaErrorStr); 
					return 1;
				}
		
				cerr = cudaMemcpyAsync(bin_desc_n_plus_x_recs_host + offset, bin_desc_n_plus_x_recs_device + offset, sizeof(uint32_t) * (BIN_NO), cudaMemcpyDeviceToHost, stream[i - 1]);
				if(cerr != cudaSuccess) {
					cudaErrorStr = cudaGetErrorString(cerr);
					printf("cannot copy bin_desc_n_plus_x_recs to host: %s\n", cudaErrorStr); 
					return 1;
				}
		
				cerr = cudaMemcpyAsync(bin_desc_n_super_kmers_host + offset, bin_desc_n_super_kmers_device + offset, sizeof(uint64_t) * (BIN_NO), cudaMemcpyDeviceToHost, stream[i - 1]);
				if(cerr != cudaSuccess) {
					cudaErrorStr = cudaGetErrorString(cerr);
					printf("cannot copy bin_desc_n_super_kmers to host: %s\n", cudaErrorStr); 
					return 1;
				}
				offset = new_offset;
				
			}
			
			cudaDeviceSynchronize();
 			readCounter = newReadCounter;

			batch_num++;
			printf("batch done\n");
		}

		putBinsToDisk(binFiles, tmp_size, binDescHost, buffer_host, bin_desc_size_host, bin_desc_n_rec_host, bin_desc_n_plus_x_recs_host, bin_desc_n_super_kmers_host
				, buffer_size_per_stream, bin_size);
		
		gettimeofday(&(timeEnd), NULL);
		timeval res;
		timersub(&(timeEnd), &(timeStart), &res);
		processReadsTimer += res.tv_sec + res.tv_usec/1000000.0; 	
		
		for(int i = 0; i < NUM_STREAM; i++) {
			cudaStreamDestroy(stream[i]);
		}
		
		//free memory after 1st stage
		cudaFree(readListDevice);
		cudaFree(binStatsDevice);
		cudaFree(sigMapDevice);
		cudaFree(buffer_device);
		cudaFree(bin_desc_size_device);
		cudaFree(bin_desc_n_rec_device);
		cudaFree(bin_desc_n_plus_x_recs_device);
		cudaFree(bin_desc_n_super_kmers_device);
		cudaFree(lock_device);
		cudaFree(norm7);
		cudaFreeHost(buffer_host);
		cudaFreeHost(bin_desc_size_host);
		cudaFreeHost(bin_desc_n_rec_host);
		cudaFreeHost(bin_desc_n_plus_x_recs_host);
		cudaFreeHost(bin_desc_n_super_kmers_host);
		
		printf("fastq reader timer is:%f \n", fastqReaderTimer); 
		printf("process read timer is:%f \n", processReadsTimer);
		cout<<"tmp size is: "<<tmp_size<<endl;
		/*
		for(int i = 0; i < BIN_NO; i++) {
			fprintf(outFile, "%d %d %d %d\n", binDescHost.bin_desc_size[i], binDescHost.bin_desc_n_rec[i], binDescHost.bin_desc_n_plus_x_recs[i],binDescHost.bin_desc_n_super_kmers[i]);
		}		
		*/
		
		//start stage 2
		printf("stage 2 started\n");

		double second_timer = 0;
		gettimeofday(&(timeStart), NULL);


		//host variables
		uint32_t max_x;
		uint64_t counter_max = 255;

		//device variables	
		//uint64_t *startArrayDevice;
		//uint64_t *posArrayDevice;


		rewindfiles(binFiles);

		/*cerr = cudaMalloc(&startArrayDevice, sizeof(uint64_t) * POS_ARRAY_SIZE);
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for start array on device\n");
			return 1;
		}

		cerr = cudaMalloc(&posArrayDevice, sizeof(uint64_t) * POS_ARRAY_SIZE);
		if(cerr != cudaSuccess) {
			printf("cannot allocate memory for pos array on device\n");
			return 1;
		}*/

		//building lookuptable
		unsigned char lut[256];
		for(int i = 0; i < 256; i++)
			lut[i] = ((3 - (i & 3)) << 6) + ((3 - ((i >> 2) & 3)) << 4) + ((3 - ((i >> 4) & 3)) << 2) + (3 - ((i >> 6) & 3));



		//calculate lut prefix len
		uint32_t best_lut_prefix_len = 0;
		uint64_t best_mem_amount = 1ull << 62;

		uint32_t lut_prefix_len;

		for (lut_prefix_len = 2; lut_prefix_len < 16; ++lut_prefix_len)
		{
			uint32_t suffix_len = KMER_LENGTH - lut_prefix_len;
			if (suffix_len % 4)
				continue;

			uint64_t est_suf_mem = n_reads * suffix_len;
			uint64_t lut_mem = BIN_NO * (1ull << (2 * lut_prefix_len)) * sizeof(uint64_t);

			if (est_suf_mem + lut_mem < best_mem_amount)
			{
				best_lut_prefix_len = lut_prefix_len;
				best_mem_amount = est_suf_mem + lut_mem;
			}
		}

		lut_prefix_len = best_lut_prefix_len;

		if(KMER_LENGTH % 32 == 0)
			max_x = 0;
		else
			max_x = MIN(31 -(KMER_LENGTH % 32), KMER_X);


		printf("kmerword is: %d\n", (int)KMER_WORDS);
		printf("n_reads is: %ld\n", n_reads);
		printf("lut prefix len is: %d\n", lut_prefix_len);

		//open file for storing result
		string kmer_file_name = "result.kmc_suf";
		string lut_file_name = "result.kmc_pre";


		FILE *out_kmer = fopen(kmer_file_name.c_str(), "wb");
		if(!out_kmer)
		{
			cout << "Error: Cannot create " << kmer_file_name << "\n";
			exit(1);
		
		}

		FILE *out_lut = fopen(lut_file_name.c_str(), "wb");
		if(!out_lut)
		{
			cout << "Error: Cannot create " << lut_file_name << "\n";
			fclose(out_kmer);
			exit(1);
		}

		char s_kmc_pre[] = "KMCP";
		char s_kmc_suf[] = "KMCS";

		// Markers at the beginning
		fwrite(s_kmc_pre, 1, 4, out_lut);
		fwrite(s_kmc_suf, 1, 4, out_kmer);


		completerDesc completer;

		uint64_t n_unique, n_cutoff_min, n_cutoff_max, n_total;
		n_unique  = n_cutoff_min  = n_cutoff_max  = n_total  = 0;

		uint32_t *sig_map = new uint32_t[MAP_SIZE];
		for(int i = 0; i < MAP_SIZE; i++)
			sig_map[i] = 0;
		uint32_t lut_pos = 0;
		uint64_t n_recs = 0;

		uint64_t data_size = 0;
		int counter = 0;
		cudaMemGetInfo(&mem_free, &mem_tot);
		std::cout << "Free memory : " << mem_free <<"total memory : "<<mem_tot << std::endl;

		
		int input_pos_sum = 0;
	

		omp_lock_t lock;
		omp_init_lock(&lock);	
		#pragma omp parallel for num_threads(1)	
		for(int i = 0; i < BIN_NO; i++) {
			//calculate sizes
			uint64_t startArrayHost[POS_ARRAY_SIZE];
			uint64_t posArrayHost[POS_ARRAY_SIZE];
			int arrayIdx = 0;

			
			if(binDescHost.bin_desc_size[i] == 0)	
				continue;
			uint64_t input_kmer_size;
			uint64_t kxmer_counter_size;

			input_kmer_size = binDescHost.bin_desc_n_plus_x_recs[i] * sizeof(CKmer);
			kxmer_counter_size = binDescHost.bin_desc_n_plus_x_recs[i] * sizeof(uint32_t);	

			uint64_t max_out_recs    = (binDescHost.bin_desc_n_rec[i]+1) / max(CUTOFF_MIN, 1);	
			
			uint64_t counter_size    = min(BYTE_LOG(CUTOFF_MAX), BYTE_LOG(counter_max));

			uint32_t kmer_symbols = KMER_LENGTH - lut_prefix_len;
			uint64_t kmer_bytes = kmer_symbols / 4;
			uint64_t out_buffer_size = max_out_recs * (kmer_bytes + counter_size);
				

			uint64_t lut_recs = 1 << (2 * lut_prefix_len);
			uint64_t lut_size = lut_recs * sizeof(uint64_t);
			
			uint64_t part1_size;
			uint64_t part2_size;

			part1_size = max(round_up_to_alignment(input_kmer_size) + round_up_to_alignment(kxmer_counter_size), round_up_to_alignment(binDescHost.bin_desc_size[i]));
			part2_size = max(round_up_to_alignment(input_kmer_size), round_up_to_alignment(out_buffer_size) + round_up_to_alignment(lut_size));
			uint64_t req_size = part1_size + part2_size;

			unsigned char* buffer = (unsigned char*)malloc((round_up_to_alignment(req_size) + ALIGNMENT) * sizeof(unsigned char));

			unsigned char* file_buffer;
			file_buffer = buffer;
			//read files
			fread(file_buffer, 1, binDescHost.bin_desc_size[i], binFiles[i]);	
			//calculate pos arrays
			uint32_t num_threads = 1;
			uint32_t thread_no = 0;
			uint64_t bytes_per_thread = (binDescHost.bin_desc_size[i] + num_threads - 1) / num_threads;	

			uint64_t start = 0;
			uint64_t pos = 0;

			uint32_t input_pos = 0;
			
			for(; pos < binDescHost.bin_desc_size[i]; pos += 1 + (file_buffer[pos] + KMER_LENGTH + 3) / 4) {
				if ((thread_no + 1) * bytes_per_thread <= pos)
				{
					startArrayHost[arrayIdx] = start;
					posArrayHost[arrayIdx] = pos;
					arrayIdx++;
					start = pos;
					++thread_no;
				}
			}
			if (start < pos)
			{	startArrayHost[arrayIdx] = start;
				posArrayHost[arrayIdx] = binDescHost.bin_desc_size[i];
				arrayIdx++;
			}
			CKmer * kmer_arrays;
			
			kmer_arrays = reinterpret_cast<CKmer *>(buffer + part1_size);
			thrust::host_vector<CKmer> H_data;  			


			for(int index = 0; index < arrayIdx; index++) {
				expandKxmers(file_buffer, kmer_arrays, startArrayHost[index], posArrayHost[index], max_x, lut, input_pos);
			}
			
			if(binDescHost.bin_desc_size[i] != 0) {
				input_pos_sum += input_pos;
				counter += posArrayHost[arrayIdx -1];
			}
			arrayIdx = 0;
	
			CKmer * result_arrays;
			result_arrays = reinterpret_cast<CKmer *>(buffer);				

			uint64_t sort_rec = input_pos;
	
			omp_set_lock(&lock);
			CKmer* d_data;
			cerr = cudaMalloc((void**)&d_data, sizeof(CKmer) * sort_rec);
			if(cerr != cudaSuccess) {
				printf("cannot allocate memory for d_data on device\n");
				exit(1);
			}

			//printf("size of d_data is: %d\n", sizeof(CKmer) * sort_rec);
			cerr = cudaMemcpy(d_data, kmer_arrays, sizeof(CKmer) * sort_rec, cudaMemcpyHostToDevice);
			if(cerr != cudaSuccess) {
				cudaErrorStr = cudaGetErrorString(cerr);
				printf("cannot d_data to device: %s\n", cudaErrorStr); 
				exit(1);
			}
		
			uint64_t *d_key;
			cerr = cudaMalloc((void**)&d_key, sizeof(uint64_t) * sort_rec);
			if(cerr != cudaSuccess) {
				printf("cannot allocate memory for d_key on device\n");
				exit(1);
			}

			num_block = ceil((double)sort_rec /(double)1024);
			
			//sort the kxmers			

			for(int sort_idx = 0; sort_idx < KMER_WORDS; sort_idx++) {
				copy_key<<<num_block, 1024>>>(d_key, d_data, sort_idx);
				thrust::device_ptr<CKmer> data_ptr(d_data);
				thrust::device_ptr<uint64_t> key_ptr(d_key);
				
				thrust::sort_by_key(key_ptr, key_ptr + sort_rec, data_ptr);
				cudaDeviceSynchronize();
			}
			cerr = cudaThreadSynchronize();
			if(cerr != cudaSuccess) {
				cudaErrorStr = cudaGetErrorString(cerr);
				printf("cuda error is: %s\n", cudaErrorStr); 
				exit(1);
			}

			cerr = cudaMemcpy(result_arrays, d_data, sizeof(CKmer) * sort_rec, cudaMemcpyDeviceToHost);
			if(cerr != cudaSuccess) {
				cudaErrorStr = cudaGetErrorString(cerr);
				printf("cannot d_data back to host: %s\n", cudaErrorStr); 
				exit(1);
			}
			cudaFree(d_data);
			cudaFree(d_key);			

			omp_unset_lock(&lock);
			
			compact_kxmers(result_arrays, lut_prefix_len, buffer, part1_size, out_buffer_size, 
				       sort_rec, input_kmer_size, max_x, counter_max, completer, i);
			
			//compact the kxmers
			data_size += completer.data_size[i];				
		
			lut_recs = completer.lut_size[i] / sizeof(uint64_t);

			omp_set_lock(&lock);
			fwrite(completer.data[i], 1, completer.data_size[i], out_kmer);

			uint64_t *ulut = (uint64_t*)completer.lut[i];
			for(uint64_t j = 0; j < lut_recs; ++j)
			{
				uint64_t x = ulut[j];
				ulut[j] = n_recs;
				n_recs += x;
			}
			fwrite(completer.lut[i], lut_recs, sizeof(uint64_t), out_lut);		
	
			omp_unset_lock(&lock);

			n_unique     += completer._n_unique[i];
			n_cutoff_min += completer._n_cutoff_min[i];
			n_cutoff_max += completer._n_cutoff_max[i];
			n_total      += completer._n_total[i];
			for (uint32_t j = 0; j < MAP_SIZE; ++j)
			{
				if (signature_map[j] == i)
				{
					sig_map[j] = lut_pos;
				}
			}
			++lut_pos;
			free(buffer); 
		}

		// Marker at the end
		fwrite(s_kmc_suf, 1, 4, out_kmer);
		fclose(out_kmer);

		fwrite(&n_recs, 1, sizeof(uint64_t), out_lut);

		//store signature mapping 
		fwrite(sig_map, sizeof(uint32_t), MAP_SIZE, out_lut);	

		uint64_t counter_size = counter_size = min(BYTE_LOG(CUTOFF_MAX), BYTE_LOG(counter_max));
		// Store header
		uint32_t offset = 0;

		store_uint(out_lut, KMER_LENGTH, 4);				offset += 4;
		store_uint(out_lut, (uint32_t)false, 4);			offset += 4;	// mode: 0 (counting), 1 (Quake-compatibile counting)
		store_uint(out_lut, counter_size, 4);				offset += 4;
		store_uint(out_lut, lut_prefix_len, 4);				offset += 4;
		store_uint(out_lut, SIGNATURE_LEN, 4);				offset += 4; 
		store_uint(out_lut, CUTOFF_MIN, 4);				offset += 4;
		store_uint(out_lut, CUTOFF_MAX, 4);				offset += 4;
		store_uint(out_lut, n_unique - n_cutoff_min - n_cutoff_max, 8);		offset += 8;

		// Space for future use
		for(int32_t i = 0; i < 7; ++i)
		{
			store_uint(out_lut, 0, 4);
			offset += 4;
		}
	
		store_uint(out_lut, 0x200, 4);
		offset += 4;

		store_uint(out_lut, offset, 4);

		// Marker at the end
		fwrite(s_kmc_pre, 1, 4, out_lut);
		fclose(out_lut);
			
		gettimeofday(&(timeEnd), NULL);
		timersub(&(timeEnd), &(timeStart), &res);
		second_timer += res.tv_sec + res.tv_usec/1000000.0; 	

		cout<<"second stage timer is: "<<second_timer<<endl;

		printf("input_pos sum is: %d\n", input_pos_sum);

		closefiles(binFiles);
		deletefiles();

		//fclose(outFile);
		printf("done\n");
		return 0;

}
