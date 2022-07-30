/***  File Header  ************************************************************/
/**
* postprocess.h
*
* system setting - used throughout the system
* @author   Shozo Fukuda
* @date     create Fri Jul 29 16:27:50 JST 2022
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
*******************************************************************************/
#ifndef _POSTPROCESS_H
#define _POSTPROCESS_H

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

/**************************************************************************}}}**
*
***************************************************************************{{{*/
std::string non_max_suppression_multi_class(unsigned int num_boxes, unsigned int box_repr, const float* boxes, unsigned int num_class, const float* scores, float iou_threshold, float score_threshold, float sigma);

#endif /* _POSTPROCESS_H */
