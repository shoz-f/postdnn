/***  File Header  ************************************************************/
/**
* post_dnn_nif.cpp
*
* Elixir/Erlang Port ext. of Post-processing for DNN.
* @author   Shozo Fukuda
* @date     create Fri Jul 29 16:27:50 JST 2022
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
**/
/**************************************************************************{{{*/

#include "my_erl_nif.h"
#include "postprocess.h"

/***  Module Header  ******************************************************}}}*/
/**
* Non Maximum Suppression for Multi Class
* @par DESCRIPTION
*   run non-maximum on every class
*
* @retval json
**/
/**************************************************************************{{{*/
DECL_NIF(non_max_suppression_multi_class) {
    ErlNifBinary boxes, scores;
    unsigned int num_boxes, box_repr;
    unsigned int num_class;
    double iou_threshold, score_threshold, sigma;
    
    if (ality != 8
    ||  !enif_get_uint(env, term[0], &num_boxes)
    ||  !enif_get_uint(env, term[1], &box_repr)
    ||  !enif_inspect_binary(env, term[2], &boxes)
    ||  !enif_get_uint(env, term[3], &num_class)
    ||  !enif_inspect_binary(env, term[4], &scores)
    ||  !enif_get_double(env, term[5], &iou_threshold)
    ||  !enif_get_double(env, term[6], &score_threshold)
    ||  !enif_get_double(env, term[7], &sigma)) {
        return enif_make_badarg(env);
    }
    
    std::string res = non_max_suppression_multi_class(
        num_boxes,
        box_repr,
        reinterpret_cast<float*>(boxes.data),
        num_class,
        reinterpret_cast<float*>(scores.data),
        iou_threshold,
        score_threshold,
        sigma
    );

    return enif_make_tuple2(env, enif_make_ok(env),
        ((!res.empty()) ? enif_make_string(env, res.c_str(), ERL_NIF_LATIN1) : enif_make_nil(env)));
}

/**************************************************************************}}}*/
/* enif resource setup                                                        */
/**************************************************************************{{{*/
int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info)
{
    return 0;
}

/**************************************************************************}}}*/
/* enif function dispach table                                                */
/**************************************************************************{{{*/
static ErlNifFunc nif_funcs[] = {
//  {erl_function_name, erl_function_arity, c_function, dirty_flags}
    #include "postdnn_nif.inc"
};

ERL_NIF_INIT(Elixir.PostDNN.NIF, nif_funcs, load, NULL, NULL, NULL)

/*** post_dnn_nif.cpp *****************************************************}}}*/
