/*
  This file is a part of KMC software distributed under GNU GPL 3 licence.
  The homepage of the KMC project is http://sun.aei.polsl.pl/kmc
  
  Authors: Sebastian Deorowicz, Agnieszka Debudaj-Grabysz, Marek Kokot
  
  Version: 2.3.0
  Date   : 2015-08-21
*/

#ifndef _MMER_H
#define _MMER_H __host__ __device__

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>


#define MIN(x,y) ((x) < (y) ? (x) : (y))
// *************************************************************************
// *************************************************************************

using namespace std;
class CMmer
{
	uint32_t str;
	uint32_t mask;
	uint32_t current_val;
	uint32_t len;
	//uint32_t norm7[1 << 14];	


public:

/*	_MMER_H CMmer() { 
		CMmer(7);
		printf("default constructor\n"); 
	};	
*/	
	_MMER_H CMmer(/*uint32_t _len*/);

	_MMER_H void insert(unsigned char symb, uint32_t* norm7){
		str <<= 2;
		str += symb;
		str &= mask;

		current_val = norm7[str];
	};

	_MMER_H uint32_t get() const {
		return current_val;
	};

	_MMER_H bool operator==(const CMmer& x) {
		return current_val == x.current_val;
	};

	_MMER_H bool operator<(const CMmer& x) {
		return current_val < x.current_val;
	};

	_MMER_H void clear() {
		str = 0;
	};

	_MMER_H bool operator<=(const CMmer& x) {
		return current_val <= x.current_val;
	};

	_MMER_H void set(const CMmer& x) {
		str = x.str;
		current_val = x.current_val;
	};

	_MMER_H void insert(const char* seq, uint32_t* norm7) {

		str = (seq[0] << 12) + (seq[1] << 10) + (seq[2] << 8) + (seq[3] << 6) + (seq[4] << 4 ) + (seq[5] << 2) + (seq[6]);

		current_val = norm7[str];
	};
	
	_MMER_H  static bool is_allowed(uint32_t mmer, uint32_t len)
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

	_MMER_H uint32_t get_rev(uint32_t mmer, uint32_t len)
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

};



//--------------------------------------------------------------------------



#endif
