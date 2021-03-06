/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */

int main(int argc, char **argv)
{
	int ch;
	::testing::InitGoogleTest(&argc, argv);

	opterr = 0;
	while((ch = getopt(argc, argv, "di:m:")) != -1)
	{
		switch(ch)
		{
			case 'd':
				nn_set_log_level(0);
				system("rm -fr tmp/*");
				break;
			case 'i':
				g_InputImagePath = optarg;
				break;
			case 'm':
				g_CaseNumber = atoi(optarg);
				break;
			default:
				break;
		}
	}
	return RUN_ALL_TESTS();
}
