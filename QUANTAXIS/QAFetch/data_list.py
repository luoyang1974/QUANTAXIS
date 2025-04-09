#
# The MIT License (MIT)
#
# Copyright (c) 2016-2021 yutiansut/QUANTAXIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
0201e20000000000 RZRQ (融资融券标的) 1000009226000000 RQ (转融券标的) a001050100000000 ST(ST股)

a001050200000000 SST (*ST)  1000023325000000 YJ(预警 SZSE) a001050d00000000 小盘股

a001050e00000000 大盘蓝筹 0201240000000000 cixin(次新股)
"""
wind_stock_list_special_id = {'rzrq': '0201e20000000000',
                              'rq': '1000009226000000',
                              'st': 'a001050100000000',
                              'sst': 'a001050200000000',
                              'yj': '1000023325000000',
                              'small': 'a001050d00000000',
                              'big': 'a001050e00000000',
                              'cixin': '0201240000000000'
                              }


HqData=[
["bk400128973","400128973","动物保健",20190816150848,7.69,7.70,8.04,1578157,198837.62,0.35,4.61,20190816150855,17,10,8,0,2,"sh600195","600195","中牧股份",16.19,340428,53216.96,1.47,9.99],
["bk400128939","400128939","旅游综合",20190816150848,27.35,27.39,28.30,964226,179228.84,0.95,3.47,20190816150853,17,15,14,0,1,"sh600138","600138","中青旅",12.18,330488,39734.78,1.00,8.94],
["bk400128937","400128937","景点",20190816150848,13.83,13.82,14.23,252853,44818.12,0.41,2.94,20190816150854,17,9,9,0,0,"sh603199","603199","九华旅游",24.43,36255,8791.88,1.39,6.03],
["bk400128924","400128924","食品加工",20190816150848,22.01,20.11,22.61,5621754,886949.35,0.60,2.71,20190816150855,17,58,38,3,17,"sz300146","300146","汤臣倍健",19.73,273610,52967.86,1.45,7.93],
["bk400128918","400128918","林业",20190816150848,5.29,5.32,5.40,192930,12421.49,0.12,2.19,20190816150843,17,3,3,0,0,"sz002679","002679","福建金森",14.74,57733,8391.10,0.73,5.21],
["bk400128922","400128922","畜禽养殖",20190816150848,25.69,25.58,26.23,2041763,551498.06,0.54,2.10,20190816150855,17,13,11,1,1,"sz002458","002458","益生股份",25.88,415987,105578.43,1.70,7.03],
["bk400128884","400128884","金属非金属新材料",20190816150848,7.81,7.73,7.95,10502092,1010877.41,0.15,1.87,20190816150852,17,42,29,3,10,"sz300127","300127","银河磁体",21.44,355651,75542.43,1.95,10.01],
["bk400128962","400128962","机场",20190816150848,27.99,28.02,28.50,784020,156243.03,0.51,1.82,20190816150852,17,4,3,0,1,"sh600009","600009","上海机场",84.84,89638,75629.66,2.77,3.38],
["bk400128930","400128930","医药商业",20190816150848,17.66,17.63,17.95,3145415,579768.51,0.29,1.63,20190816150854,17,60,44,2,14,"sh603658","603658","安图生物",82.98,27428,22535.42,4.53,5.77],
["bk400128919","400128919","饲料",20190816150848,14.69,14.71,14.91,3739900,517565.84,0.23,1.54,20190816150846,17,13,10,1,2,"sz000048","000048","*ST康达",20.95,2007,420.44,1.00,5.01],
["bk400128933","400128933","一般零售",20190816150848,6.90,6.95,7.01,2678616,206958.59,0.11,1.53,20190816150855,17,56,37,7,12,"sz000759","000759","中百集团",6.90,75416,4982.62,0.61,9.70],
["bk400128946","400128946","其他电子",20190816150848,17.52,17.48,17.77,7861109,1585491.74,0.25,1.40,20190816150851,17,67,37,4,26,"sz002881","002881","美格智能",39.83,111129,42885.27,3.62,10.00],
["bk400128887","400128887","其他建材",20190816150848,7.62,7.40,7.72,1990034,192703.08,0.09,1.25,20190816150853,17,50,28,8,14,"sz000786","000786","北新建材",17.30,141311,24139.22,0.90,5.49],
["bk400128944","400128944","光学光电子",20190816150848,5.74,5.73,5.81,21733036,1505604.24,0.07,1.19,20190816150855,17,55,25,4,26,"sz002273","002273","水晶光电",12.27,682087,82104.30,0.99,8.78],
["bk400128916","400128916","种植业",20190816150848,7.09,7.12,7.17,2227745,201890.06,0.08,1.13,20190816150854,17,13,12,0,1,"sh600371","600371","万向德农",14.51,471056,67477.44,0.74,5.37],
["bk400128929","400128929","生物制品",20190816150848,17.85,17.83,18.04,3702573,701349.43,0.20,1.11,20190816150850,17,41,28,0,13,"sz300482","300482","万孚生物",45.10,34514,15439.48,2.00,4.64],
["bk400128923","400128923","饮料制造",20190816150848,82.14,82.43,83.03,3001781,1491782.60,0.89,1.09,20190816150855,17,41,32,2,7,"sh600600","600600","青岛啤酒",50.66,194819,96672.10,4.61,10.01],
["bk400128879","400128879","橡胶",20190816150848,7.16,7.15,7.23,1113379,92865.27,0.08,1.07,20190816150855,17,24,14,0,10,"sz002886","002886","沃特股份",20.19,23771,4689.97,1.84,10.03],
["bk400128904","400128904","地面兵装",20190816150848,11.20,11.21,11.32,566002,79990.13,0.12,1.03,20190816150846,17,8,7,1,0,"sz300397","300397","天和防务",20.68,196370,39619.80,1.88,10.00],
["bk400128948","400128948","计算机应用",20190816150848,12.42,12.39,12.54,17250294,2631248.69,0.12,0.97,20190816150855,17,150,102,6,42,"sz300609","300609","汇纳科技",36.88,5045,1860.48,3.35,9.99],
["bk400128927","400128927","化学制药",20190816150848,15.98,15.96,16.13,7257433,1026684.32,0.15,0.95,20190816150855,17,81,52,1,28,"sh600829","600829","人民同泰",7.57,267837,20104.57,0.69,10.03],
["bk400128947","400128947","计算机设备",20190816150848,11.98,11.96,12.09,6644186,925281.67,0.11,0.91,20190816150854,17,62,36,1,24,"sz300270","300270","中威电子",8.31,88013,7167.29,0.76,10.07],
["bk400128970","400128970","保险",20190816150848,41.29,40.73,41.65,1597714,1002762.05,0.36,0.88,20190816150850,17,6,4,1,1,"sh601318","601318","中国平安",87.46,932312,814894.31,1.15,1.33],
["bk400128951","400128951","互联网传媒",20190816150848,8.72,8.59,8.79,8950734,1034699.22,0.07,0.85,20190816150850,17,61,41,4,16,"sz002602","002602","世纪华通",8.79,235693,20571.23,0.28,3.29],
["bk400128893","400128893","电机",20190816150848,6.30,6.34,6.35,476630,50206.55,0.05,0.84,20190816150855,17,12,7,2,3,"sz002892","002892","科力尔",47.49,46534,21374.05,3.83,8.77],
["bk400128899","400128899","仪器仪表",20190816150848,9.77,9.76,9.85,1186684,199450.79,0.08,0.81,20190816150854,17,30,24,1,5,"sz002870","002870","香山股份",21.37,44176,9468.05,0.54,2.59],
["bk400128903","400128903","航空装备",20190816150848,14.21,14.21,14.31,1825389,313380.16,0.10,0.71,20190816150848,17,21,15,1,5,"sz002190","002190","*ST集成",17.95,27566,4910.20,0.65,3.76],
["bk400128885","400128885","水泥制造",20190816150848,14.88,14.88,14.98,1464609,177834.50,0.10,0.68,20190816150853,17,17,15,0,2,"sh600720","600720","祁连山",8.70,255763,22084.73,0.26,3.08],
["bk400128964","400128964","铁路运输",20190816150848,6.31,6.31,6.36,570851,51967.96,0.04,0.67,20190816150848,17,7,4,0,3,"sz002800","002800","天顺股份",22.10,71294,15282.56,2.01,10.00],
["bk400128942","400128942","半导体",20190816150848,15.68,15.63,15.78,5189389,969638.38,0.11,0.67,20190816150853,17,32,12,0,20,"sz002180","002180","纳思达",23.96,161933,37656.65,2.18,10.01],
["bk400128949","400128949","文化传媒",20190816150848,6.95,6.95,7.00,5721706,448560.91,0.04,0.62,20190816150855,17,79,48,5,26,"sz000665","000665","湖北广电",5.58,156458,8573.08,0.51,10.06],
["bk400128913","400128913","包装印刷",20190816150848,7.39,7.37,7.43,2354994,176737.89,0.04,0.60,20190816150854,17,31,16,1,14,"sz002243","002243","通产丽星",13.18,150603,19445.43,0.35,2.73],
["bk400128971","400128971","多元金融",20190816150848,6.05,6.05,6.09,3039112,190220.13,0.03,0.57,20190816150850,17,24,17,5,2,"sz000416","000416","民生控股",5.65,331157,19364.75,0.24,4.44],
["bk400128928","400128928","中药",20190816150848,12.23,12.22,12.30,6636472,714724.83,0.07,0.56,20190816150855,17,60,30,4,26,"sz000650","000650","仁和药业",7.15,495942,35322.42,0.24,3.47],
["bk400128902","400128902","航天装备",20190816150848,9.95,9.95,10.00,877440,103069.53,0.05,0.55,20190816150855,17,14,10,1,3,"sh600677","600677","航天通信",13.67,209792,28747.80,0.14,1.03],
["bk400128882","400128882","黄金",20190816150848,8.75,8.81,8.80,4935355,573993.69,0.05,0.55,20190816150855,17,11,7,2,2,"sh600547","600547","山东黄金",51.25,413361,211118.15,0.66,1.30],
["bk400128959","400128959","高速公路",20190816150848,5.13,5.13,5.15,849711,38278.57,0.03,0.50,20190816150853,17,19,11,6,2,"sh600106","600106","重庆路桥",2.99,129419,3878.53,0.07,2.40],
["bk400128883","400128883","稀有金属",20190816150848,8.32,8.35,8.36,4968517,584186.65,0.04,0.50,20190816150846,17,21,12,0,9,"sz000831","000831","五矿稀土",14.33,604242,85848.30,0.67,4.90],
["bk400128938","400128938","酒店",20190816150848,9.81,9.86,9.86,95242,7380.88,0.05,0.48,20190816150852,17,7,5,1,1,"sz200613","200613","大东海B",2.87,403,11.50,0.07,2.50],
["bk400128914","400128914","家用轻工",20190816150848,7.46,7.47,7.50,3871279,298488.13,0.03,0.47,20190816150855,17,68,40,2,26,"sz000910","000910","大亚圣象",10.72,80074,8363.16,0.84,8.50],
["bk400128957","400128957","环保工程及服务",20190816150848,6.39,6.31,6.42,4071068,280036.78,0.03,0.47,20190816150848,17,49,22,5,22,"sh600217","600217","中再资环",5.55,283011,15649.67,0.27,5.11],
["bk400128909","400128909","其他交运设备",20190816150848,5.27,5.26,5.30,380604,25969.82,0.02,0.45,20190816150848,17,12,7,0,5,"sh600877","600877","*ST嘉陵",6.35,58118,3692.47,0.18,2.92],
["bk400128876","400128876","化学制品",20190816150848,8.74,8.73,8.77,13733639,1539184.52,0.04,0.43,20190816150855,17,192,105,5,82,"sz002201","002201","九鼎新材",16.70,375662,61293.31,1.52,10.01],
["bk400128881","400128881","工业金属",20190816150848,4.65,4.66,4.67,6194712,372582.89,0.02,0.41,20190816150855,17,43,20,6,17,"sh603115","603115","海星股份",23.61,4605,1087.17,2.15,10.02],
["bk400128920","400128920","农产品加工",20190816150848,8.57,8.47,8.60,2184406,306561.41,0.03,0.40,20190816150855,17,22,12,2,8,"sz002234","002234","民和股份",36.52,349232,123621.44,3.32,10.00],
["bk400128886","400128886","玻璃制造",20190816150848,4.21,4.21,4.23,482610,33066.27,0.02,0.38,20190816150853,17,12,8,2,2,"sh900918","900918","耀皮Ｂ股",0.41,1903,7.73,0.01,3.80],
["bk400128969","400128969","证券",20190816150848,9.88,9.89,9.91,12702718,1438306.91,0.03,0.35,20190816150855,17,40,26,6,8,"sh601236","601236","红塔证券",14.73,1532156,224422.10,0.77,5.52],
["bk400128972","400128972","综合",20190816150848,4.40,4.39,4.42,4015547,206584.76,0.02,0.35,20190816150855,17,47,28,5,14,"sh600421","600421","*ST仰帆",8.60,11587,994.45,0.41,5.01],
["bk400128917","400128917","渔业",20190816150848,3.71,3.70,3.72,673925,27228.83,0.01,0.35,20190816150845,17,11,8,1,2,"sz002086","002086","ST东海洋",3.26,102399,3351.26,0.10,3.16],
["bk400128925","400128925","纺织制造",20190816150848,4.86,4.85,4.87,1336727,79111.51,0.01,0.29,20190816150854,17,37,20,4,13,"sh600152","600152","维科技术",6.52,337968,21914.51,0.59,9.95],
["bk400128954","400128954","电力",20190816150848,6.24,6.25,6.26,5872733,316979.77,0.02,0.28,20190816150855,17,71,34,8,29,"sz001896","001896","豫能控股",3.31,163408,5384.35,0.07,2.16],
["bk400128910","400128910","白色家电",20190816150848,21.68,21.65,21.74,3127554,493254.24,0.06,0.28,20190816150845,17,55,30,5,20,"sz002681","002681","奋达科技",4.33,218924,9333.62,0.14,3.34],
["bk400128931","400128931","医疗器械",20190816150848,12.95,12.87,12.98,1746015,250286.55,0.03,0.27,20190816150853,17,37,25,1,11,"sz300314","300314","戴维医疗",9.25,74118,6596.84,0.84,9.99],
["bk400128955","400128955","水务",20190816150848,5.15,5.15,5.16,429373,24499.52,0.01,0.27,20190816150850,17,17,10,3,4,"sh601158","601158","重庆水务",5.56,30037,1668.79,0.05,0.91],
["bk400128945","400128945","电子制造",20190816150848,15.45,15.38,15.49,12896489,2073380.67,0.04,0.24,20190816150853,17,61,34,2,25,"sz002402","002402","和而泰",10.34,659355,66399.01,0.94,10.00],
["bk400128889","400128889","装修装饰",20190816150848,6.70,6.70,6.72,1260700,86183.59,0.02,0.23,20190816150854,17,28,14,3,11,"sz002375","002375","亚厦股份",5.41,98208,5399.17,0.15,2.85],
["bk400128960","400128960","公交",20190816150848,4.56,4.56,4.57,213889,10578.59,0.01,0.23,20190816150855,17,9,7,1,1,"sh600662","600662","强生控股",4.67,44812,2093.72,0.06,1.30],
["bk400128932","400128932","医疗服务",20190816150848,16.47,16.51,16.51,1602340,275010.64,0.03,0.21,20190816150848,17,22,11,1,10,"sz300642","300642","透景生命",38.65,9910,3820.09,0.82,2.17],
["bk400128897","400128897","通用机械",20190816150848,7.70,7.70,7.71,6848032,684862.35,0.02,0.20,20190816150855,17,134,66,13,55,"sz002884","002884","凌霄泵业",14.15,51958,7175.67,1.29,10.03],
["bk400128965","400128965","物流",20190816150848,8.10,8.08,8.12,2277637,184284.10,0.01,0.15,20190816150853,17,36,21,2,13,"sz300538","300538","同益股份",18.58,77009,14433.75,0.57,3.16],
["bk400128906","400128906","汽车整车",20190816150848,11.41,11.44,11.42,2131716,206299.86,0.01,0.12,20190816150853,17,25,15,2,8,"sh600686","600686","金龙汽车",7.39,74084,5502.47,0.21,2.92],
["bk400128878","400128878","塑料",20190816150848,5.09,4.51,5.09,1581314,95522.59,0.01,0.12,20190816150852,17,28,12,4,12,"sz002263","002263","*ST东南",1.92,144132,2749.01,0.06,3.23],
["bk400128871","400128871","煤炭开采",20190816150848,7.11,7.12,7.12,4401104,264814.73,0.01,0.09,20190816150855,17,37,16,8,13,"sh900948","900948","伊泰Ｂ股",0.90,36125,326.28,0.04,4.76],
["bk400128912","400128912","造纸",20190816150848,5.15,5.15,5.15,1770176,145583.04,0.00,0.08,20190816150854,17,26,11,3,12,"sz002521","002521","齐峰新材",5.49,102240,5593.24,0.20,3.78],
["bk400128872","400128872","其他采掘",20190816150848,4.24,4.23,4.24,579704,32726.16,0.00,0.06,20190816150854,17,10,7,0,3,"sh600193","600193","ST创兴",3.79,22641,858.48,0.07,1.88],
["bk400128921","400128921","农业综合",20190816150848,4.39,4.40,4.39,863883,38046.18,0.00,0.06,20190816150848,17,7,5,1,1,"sh600251","600251","冠农股份",5.31,68065,3595.47,0.15,2.91],
["bk400128956","400128956","燃气",20190816150848,6.28,6.27,6.28,639174,42001.08,0.00,0.06,20190816150854,17,20,9,1,10,"sz000669","000669","金鸿控股",4.25,112340,4773.90,0.39,10.10],
["bk400128892","400128892","园林工程",20190816150848,4.27,4.15,4.27,662547,35756.51,0.00,0.05,20190816150854,17,20,10,5,5,"sz002775","002775","文科园林",5.71,18869,1074.65,0.11,1.96],
["bk400128891","400128891","专业工程",20190816150848,4.47,4.47,4.47,1363724,77269.75,0.00,0.04,20190816150854,17,30,19,4,7,"sh603698","603698","航天工程",11.97,12811,1527.21,0.14,1.18],
["bk400128953","400128953","通信设备",20190816150848,11.65,11.64,11.65,11569996,1518136.97,0.00,0.02,20190816150855,17,91,47,3,41,"sz002194","002194","武汉凡谷",20.11,14855,2987.42,1.83,10.01],
["bk400128963","400128963","航运",20190816150848,3.99,4.00,3.99,1355526,52414.92,0.00,0.00,20190816150845,17,10,5,2,3,"sh601919","601919","中远海控",4.45,111546,4930.84,0.06,1.37],
["bk400128898","400128898","专用设备",20190816150848,7.47,7.47,7.47,9100937,992384.78,0.00,0.00,20190816150855,17,147,86,10,51,"sh603617","603617","君禾股份",19.13,226504,45186.35,-2.13,-10.02],
["bk400128875","400128875","化学原料",20190816150848,4.70,4.70,4.70,3104316,231239.27,0.00,-0.01,20190816150852,17,51,26,7,18,"sh603790","603790","雅运股份",15.92,47310,7554.08,-0.45,-2.75],
["bk400128900","400128900","金属制品",20190816150848,5.57,5.57,5.57,910281,54503.77,0.00,-0.01,20190816150853,17,20,10,2,8,"sz002487","002487","大金重工",4.05,30347,1238.13,-0.05,-1.22],
["bk400128870","400128870","石油开采",20190816150848,5.99,5.98,5.99,914488,46111.04,0.00,-0.02,20190816150841,17,6,0,1,5,"sh600759","600759","洲际油气",3.06,189245,5831.34,-0.03,-0.97],
["bk400128874","400128874","石油化工",20190816150848,4.92,4.91,4.92,1419666,73659.90,0.00,-0.04,20190816150855,17,18,8,3,7,"sh600256","600256","广汇能源",3.30,255693,8473.99,-0.04,-1.20],
["bk400128935","400128935","商业物业经营",20190816150848,4.22,4.22,4.22,539290,29408.35,0.00,-0.04,20190816150852,17,15,6,4,5,"sz200058","200058","深赛格B",2.29,867,19.96,-0.03,-1.29],
["bk400128926","400128926","服装家纺",20190816150848,6.29,6.30,6.29,2543740,171476.34,0.00,-0.06,20190816150855,17,56,32,4,20,"sz300526","300526","中潜股份",37.16,60819,23784.58,-3.13,-7.77],
["bk400128966","400128966","房地产开发",20190816150848,7.26,7.22,7.26,12180829,906957.71,0.00,-0.06,20190816150855,17,133,59,20,53,"sz000897","000897","*ST津滨",2.17,132586,2880.60,-0.11,-4.82],
["bk400128907","400128907","汽车零部件",20190816150848,8.34,8.33,8.33,4936033,450298.91,-0.01,-0.08,20190816150855,17,132,75,12,45,"sz000760","000760","*ST斯太",1.46,275706,4079.08,-0.08,-5.19],
["bk400128873","400128873","采掘服务",20190816150848,4.50,4.51,4.50,637961,63977.78,0.00,-0.10,20190816150850,17,13,6,1,6,"sh603619","603619","中曼石油",15.33,36038,5582.51,-0.15,-0.97],
["bk400128940","400128940","餐饮",20190816150848,4.56,4.56,4.55,55644,2313.57,-0.01,-0.14,20190816150803,17,3,1,1,1,"sz002306","002306","ST云网",2.72,14103,386.62,-0.02,-0.73],
["bk400128934","400128934","专业零售",20190816150848,6.72,6.73,6.71,426523,39429.48,-0.01,-0.15,20190816150852,17,8,3,1,4,"sz300022","300022","吉峰科技",3.86,68805,2673.61,-0.04,-1.03],
["bk400128890","400128890","基础建设",20190816150848,6.73,6.71,6.72,2648396,232240.49,-0.01,-0.16,20190816150855,17,35,14,6,15,"sh603909","603909","合诚股份",16.72,13504,2280.86,-0.44,-2.56],
["bk400128936","400128936","贸易",20190816150848,6.89,6.87,6.88,1128835,82671.96,-0.01,-0.16,20190816150855,17,27,14,1,12,"sz002127","002127","南极电商",9.58,147232,14101.53,-0.23,-2.34],
["bk400128894","400128894","电气自动化设备",20190816150848,8.40,8.40,8.38,2518971,186161.52,-0.01,-0.17,20190816150852,17,44,24,1,19,"sz300208","300208","青岛中程",8.16,39811,3299.42,-0.22,-2.63],
["bk400128908","400128908","汽车服务",20190816150848,3.92,3.92,3.91,775981,42252.36,-0.01,-0.20,20190816150855,17,15,8,2,5,"sh600653","600653","申华控股",2.08,39384,823.38,-0.02,-0.95],
["bk400128915","400128915","其他轻工制造",20190816150848,10.18,10.21,10.16,567404,106818.01,-0.03,-0.26,20190816150848,17,9,5,1,3,"sz002812","002812","恩捷股份",30.09,34766,10481.55,-0.62,-2.02],
["bk400128880","400128880","钢铁",20190816150848,3.30,3.30,3.29,4055713,147383.19,-0.01,-0.28,20190816150854,17,34,8,4,22,"sh601003","601003","柳钢股份",4.94,160398,7971.30,-0.14,-2.76],
["bk400128958","400128958","港口",20190816150848,4.16,4.15,4.14,1725377,79485.71,-0.01,-0.30,20190816150855,17,22,7,7,8,"sh600575","600575","皖江物流",2.54,241239,6168.05,-0.04,-1.55],
["bk400128961","400128961","航空运输",20190816150848,5.91,5.91,5.89,1581705,146367.67,-0.02,-0.31,20190816150855,17,13,4,0,9,"sh603871","603871","嘉友国际",32.62,20128,6625.92,-1.08,-3.20],
["bk400128895","400128895","电源设备",20190816150848,7.05,7.06,7.02,6532399,586579.15,-0.02,-0.31,20190816150850,17,79,41,2,36,"sz300153","300153","科泰电源",6.53,191008,12466.54,-0.19,-2.83],
["bk400128888","400128888","房屋建设",20190816150848,5.06,5.07,5.05,1195221,65342.29,-0.02,-0.32,20190816150854,17,12,4,1,7,"sz200018","200018","*ST神城B",0.69,2351,16.17,-0.02,-2.82],
["bk400128911","400128911","视听器材",20190816150848,3.73,3.71,3.72,2754756,118387.69,-0.01,-0.33,20190816150845,17,15,7,0,8,"sz002848","002848","高斯贝尔",13.86,112223,15882.51,-0.30,-2.12],
["bk400128968","400128968","银行",20190816150848,5.79,5.79,5.77,12039469,1017630.74,-0.02,-0.40,20190816150854,17,33,11,1,21,"sh601398","601398","工商银行",5.50,1152928,63779.13,-0.06,-1.08],
["bk400128941","400128941","其他休闲服务",20190816150848,5.21,5.22,5.19,119091,8626.58,-0.02,-0.47,20190816150803,17,5,3,0,2,"sz000863","000863","三湘印象",4.81,11815,570.65,-0.06,-1.23],
["bk400128967","400128967","园区开发",20190816150848,16.67,16.67,16.59,293708,24688.50,-0.08,-0.50,20190816150852,17,4,2,0,2,"sz001979","001979","招商蛇口",19.43,56280,11008.05,-0.11,-0.56],
["bk400128943","400128943","元件",20190816150848,14.35,14.26,14.26,5720579,786122.19,-0.09,-0.63,20190816150855,17,31,12,0,19,"sz300408","300408","三环集团",19.50,136688,26713.28,-0.65,-3.23],
["bk400128901","400128901","运输设备",20190816150848,6.73,6.73,6.69,678224,44000.82,-0.04,-0.64,20190816150852,17,9,6,0,3,"sz000925","000925","众合科技",6.30,40765,2582.97,-0.08,-1.25],
["bk400128950","400128950","营销传播",20190816150848,5.39,5.40,5.36,1932321,115885.65,-0.04,-0.67,20190816150850,17,20,5,1,14,"sz300612","300612","宣亚国际",18.71,28379,5328.85,-0.69,-3.56]
]