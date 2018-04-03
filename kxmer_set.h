/*
  This file is a part of KMC software distributed under GNU GPL 3 licence.
  The homepage of the KMC project is http://sun.aei.polsl.pl/kmc
  
  Authors: Sebastian Deorowicz, Agnieszka Debudaj-Grabysz, Marek Kokot
  
  Version: 2.0
  Date   : 2014-07-04
*/
#ifndef _KXMER_SET_
#define _KXMER_SET_
#include "kmer.h"

using namespace std;

#define KXMER_SET_SIZE 1024 //KMC_2: temporarily



class CKXmerSet
{
	typedef pair<pair<uint64_t, uint64_t>, uint32_t> elem_desc_t; //start_pos, end_pos, shr
	typedef pair<CKmer, uint32_t> heap_elem_t; //kxmer val, desc_id
	elem_desc_t data_desc[KXMER_SET_SIZE];
	heap_elem_t data[KXMER_SET_SIZE];
	uint32_t pos;
	uint32_t desc_pos;
	CKmer mask;

	CKmer* buffer;

	inline void update_heap()
	{
		uint32_t desc_id = data[1].second;
		CKmer kmer;
		if (++(data_desc[desc_id]).first.first < (data_desc[desc_id]).first.second)
		{
			kmer.from_kxmer(buffer[(data_desc[desc_id]).first.first], (data_desc[desc_id]).second, mask);
		}
		else
		{
			kmer.set(data[--pos].first);
			desc_id = data[pos].second;
		}
		
		uint32_t parent, less;
		parent = less = 1;
		while (true)
		{
			if (parent * 2 >= pos)
				break;
			if (parent * 2 + 1 >= pos)
				less = parent * 2;
			else if (data[parent * 2].first < data[parent * 2 + 1].first)
				less = parent * 2;
			else
				less = parent * 2 + 1;
			if (data[less].first < kmer)
			{
				data[parent] = data[less];
				parent = less;
			}			
			else
				break;
		}
		data[parent] = make_pair(kmer, desc_id);
	}

public:
	CKXmerSet(uint32_t kmer_len)
	{
		pos = 1;
		mask.set_n_1(kmer_len * 2);
		desc_pos = 0;
	}
	inline void init_add(uint64_t start_pos, uint64_t end_pos, uint32_t shr)
	{
		data_desc[desc_pos].first.first = start_pos;
		data_desc[desc_pos].first.second = end_pos;
		data_desc[desc_pos].second = shr;
		data[pos].first.from_kxmer(buffer[start_pos], shr, mask);
		data[pos].second = desc_pos;
		uint32_t child_pos = pos++;

		while (child_pos > 1 && data[child_pos].first < data[child_pos / 2].first)
		{
			swap(data[child_pos], data[child_pos / 2]);
			child_pos /= 2;
		}
		++desc_pos;
	}
	inline void set_buffer(CKmer* _buffer)
	{
		buffer = _buffer;
	}
	inline void clear()
	{
		pos = 1;
		desc_pos = 0;
	}

	inline bool get_min(uint64_t& _pos, CKmer& kmer)
	{
		if (pos <= 1)
			return false;

		kmer = data[1].first;
		_pos = (data_desc[data[1].second]).first.first;
		update_heap();
		
	
		return true;
	}
};



#endif
