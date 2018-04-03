/*
  This file is a part of KMC software distributed under GNU GPL 3 licence.
  The homepage of the KMC project is http://sun.aei.polsl.pl/kmc
  
  Authors: Sebastian Deorowicz, Agnieszka Debudaj-Grabysz, Marek Kokot
  
  Version: 2.3.0
  Date   : 2015-08-21
*/

#include "mmer.h"


//--------------------------------------------------------------------------
CMmer::CMmer(/*uint32_t _len*/)
{
	//printf("constructor being called\n");
	/*uint32_t special = 1 << 7 * 2;
	for(uint32_t i = 0 ; i < special ; ++i)
	{				
		uint32_t rev = get_rev(i, 7);
		uint32_t str_val = is_allowed(i, 7) ? i : special;
		uint32_t rev_val = is_allowed(rev, 7) ? rev : special;
		norm7[i] = MIN(str_val, rev_val);				
	}
	*/
	len = 7;
	mask = (1 << 7 * 2) - 1;
	str = 0;
}



