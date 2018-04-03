#ifndef _KMER_H
#define _KMER_H __host__ __device__

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#define SIZE 2

 struct CKmer {
	unsigned long long data[SIZE];

	static uint32_t quality_size;
	
	typedef uint64_t data_t;

	_KMER_H void set(const CKmer &x);
	
	_KMER_H void from_kxmer(const CKmer& x, uint32_t _shr, const CKmer& _mask);


	_KMER_H void mask(const CKmer &x);
	_KMER_H uint32_t end_mask(const uint32_t mask);
	_KMER_H void set_2bits(const uint64_t x, const uint32_t p);
	_KMER_H unsigned char get_2bits(const uint32_t p);
	_KMER_H unsigned char get_byte(const uint32_t p);
	_KMER_H void set_byte(const uint32_t p, unsigned char x);
	_KMER_H void set_bits(const uint32_t p, const uint32_t n, uint64_t x);

	_KMER_H void SHL_insert_2bits(const uint64_t x);
	_KMER_H void SHR_insert_2bits(const uint64_t x, const uint32_t p);

	_KMER_H void SHR(const uint32_t p);
	_KMER_H void SHL(const uint32_t p);	

	_KMER_H uint64_t remove_suffix(const uint32_t n) const;
	_KMER_H void set_n_1(const uint32_t n);
	_KMER_H void set_n_01(const uint32_t n);

	_KMER_H void store(unsigned char *&buffer, int n);
	_KMER_H void store(unsigned char *buffer, int p, int n);
	_KMER_H void load(unsigned char *&buffer, int n);

	_KMER_H bool operator==(const CKmer &x);
	_KMER_H bool operator<(const CKmer &x);

	_KMER_H void clear(void);

	_KMER_H unsigned char get_symbol(int p);
};


// *********************************************************************
 _KMER_H void CKmer::set(const CKmer &x)
{
	for(uint32_t i = 0; i < SIZE; ++i)
		data[i] = x.data[i];

}
// *********************************************************************
 _KMER_H void CKmer::from_kxmer(const CKmer& x, uint32_t _shr, const CKmer& _mask)
{
	if (_shr)
	{
		for (uint32_t i = 0; i < SIZE - 1; ++i)
		{
			data[i] = x.data[i] >> (2 * _shr);
			data[i] += x.data[i+1]<<(64-2*_shr);
		}	
		data[SIZE - 1] = x.data[SIZE - 1] >> (2 * _shr);
	}
	else
	{
		for (uint32_t i = 0; i < SIZE; ++i)
			data[i] = x.data[i];
	}
	mask(_mask);
}

// *********************************************************************
 _KMER_H void CKmer::mask(const CKmer &x) 
{
	for(uint32_t i = 0; i < SIZE; ++i)
		data[i] &= x.data[i];
}

// *********************************************************************
 _KMER_H uint32_t CKmer::end_mask(const uint32_t mask)
{
	return data[0] & mask;
}
// *********************************************************************
 _KMER_H void CKmer::set_2bits(const uint64_t x, const uint32_t p) 
{
	data[p >> 6] += x << (p & 63);
}

 _KMER_H unsigned char CKmer::get_2bits(const uint32_t p)
{
	return (data[p >> 6] >> (p & 63)) & 3;
}
// *********************************************************************
 _KMER_H void CKmer::SHR_insert_2bits(const uint64_t x, const uint32_t p) 
{
	for(uint32_t i = 0; i < SIZE-1; ++i)
	{
		data[i] >>= 2;
//		data[i] |= data[i+1] << (64-2);
		data[i] += data[i+1] << (64-2);
	}
	data[SIZE-1] >>= 2;
	data[p >> 6] += x << (p & 63);
}



// *********************************************************************
 _KMER_H void CKmer::SHR(const uint32_t p)
{
	for(uint32_t i = 0; i < SIZE-1; ++i)
	{
		data[i] >>= 2*p;
//		data[i] |= data[i+1] << (64-2*p);
		data[i] += data[i+1] << (64-2*p);
	}
	data[SIZE-1] >>= 2*p;
}


// *********************************************************************
  _KMER_H void CKmer::SHL(const uint32_t p)
{
	for(uint32_t i = SIZE-1; i > 0; --i)
	{
		data[i] <<= p*2;
//		data[i] |= data[i-1] >> (64-p*2);
		data[i] += data[i-1] >> (64-p*2);
	}
	data[0] <<= p*2;
}

// *********************************************************************
 _KMER_H  void CKmer::SHL_insert_2bits(const uint64_t x) 
{
	for(uint32_t i = SIZE-1; i > 0; --i)
	{
		data[i] <<= 2;
//		data[i] |= data[i-1] >> (64-2);
		data[i] += data[i-1] >> (64-2);
	}
	data[0] <<= 2;
//	data[0] |= x;
	data[0] += x;
}


// *********************************************************************
  _KMER_H unsigned char CKmer::get_byte(const uint32_t p) 
{
	return (data[p >> 3] >> ((p << 3) & 63)) & 0xFF; 
}

// *********************************************************************
 _KMER_H  void CKmer::set_byte(const uint32_t p, unsigned char x) 
{
//	data[p >> 3] |= ((uint64_t) x) << ((p & 7) << 3);
	data[p >> 3] += ((uint64_t) x) << ((p & 7) << 3);
}

// *********************************************************************
 _KMER_H  void CKmer::set_bits(const uint32_t p, const uint32_t n, uint64_t x)
{
//	data[p >> 6] |= x << (p & 63);
	data[p >> 6] += x << (p & 63);
	if((p >> 6) != ((p+n-1) >> 6))
//		data[(p >> 6) + 1] |= x >> (64 - (p & 63));
		data[(p >> 6) + 1] += x >> (64 - (p & 63));
}

// *********************************************************************
 _KMER_H  bool CKmer::operator==(const CKmer &x) {
	for(uint32_t i = 0; i < SIZE; ++i)
		if(data[i] != x.data[i])
			return false;

	return true;
}

// *********************************************************************
  _KMER_H bool CKmer::operator<(const CKmer &x) {
	for(int i = SIZE-1; i >= 0; --i)
		if(data[i] < x.data[i])
			return true;
		else if(data[i] > x.data[i])
			return false;
	return false;
}


// *********************************************************************
  _KMER_H void CKmer::clear(void)
{
	for(uint32_t i = 0; i < SIZE; ++i)
		data[i] = 0;
}

// *********************************************************************
  _KMER_H uint64_t CKmer::remove_suffix(const uint32_t n) const
{
	uint32_t p = n >> 6; // / 64;
	uint32_t r = n & 63;	// % 64;

	if(p == SIZE-1)
		return data[p] >> r;
	else
//		return (data[p+1] << (64-r)) | (data[p] >> r);
		return (data[p+1] << (64-r)) + (data[p] >> r);
}

// *********************************************************************
  _KMER_H void CKmer::set_n_1(const uint32_t n)
{
	clear();

	for(uint32_t i = 0; i < (n >> 6); ++i)
		data[i] = ~((uint64_t) 0);

	uint32_t r = n & 63;
	
	if(r)
		data[n >> 6] = (1ull << r) - 1;
}



// *********************************************************************
 _KMER_H  void CKmer::set_n_01(const uint32_t n)
{
	clear();

	for(uint32_t i = 0; i < n; ++i)
		if(!(i & 1))
//			data[i >> 6] |= (1ull << (i & 63));
			data[i >> 6] += (1ull << (i & 63));
}

// *********************************************************************
  _KMER_H void CKmer::store(unsigned char *&buffer, int n)
{
	for(int i = n-1; i >= 0; --i)
		*buffer++ = get_byte(i);
}

// *********************************************************************
  _KMER_H void CKmer::store(unsigned char *buffer, int p, int n)
{
	for(int i = n-1; i >= 0; --i)
		buffer[p++] = get_byte(i);
}

// *********************************************************************
  _KMER_H void CKmer::load(unsigned char *&buffer, int n)
{
	clear();
	for(int i = n-1; i >= 0; --i)
		set_byte(i, *buffer++);
}

// *********************************************************************
  _KMER_H unsigned char CKmer::get_symbol(int p)
{
	uint32_t x = (data[p >> 5] >> (2*(p & 31))) & 0x03;

	switch(x)
	{
	case 0 : return 'A';
	case 1 : return 'C';
	case 2 : return 'G';
	default: return 'T';
	}
}


#endif
