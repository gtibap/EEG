blinks_components_dict = {
    100:{
        'session_0':{
            'baseline':[[0,1,2]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
            # 'a_closed_eyes':[[],[],[],],
            # 'a_opened_eyes':[[0],[0,1],[0],],
            # 'b_closed_eyes':[[],[],[18],],
            # 'b_opened_eyes':[[0,6],[0],[0,7],],
        },
    },
    101:{
        'session_0':{
            'baseline':[[0,1,2]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    102:{
        'session_0':{
            'baseline':[[]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[8],[3,21],[6],],
        },
    },
    103:{
        'session_0':{
            'baseline':[[1, 4]],
            ## ICA components are showing that occipital (head's back) right region show very intense activity, maybe related to artifacts
            ## very few components (3-5)... which could indicate high influence of artefacts
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[1],[1, 4],[1],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[2],[2],[2],],
        },
    },
    105:{
        'session_0':{
            ## TP7 highly afffected by artifacts
            ## blinking and lateral eyes movements
            # 'ica_var': 0.999,
            'baseline':[[0,1,4,6]],
            'a_closed_eyes':[[],[8,14],[1,10],],
            'a_opened_eyes':[[0,1],[0,1],[0,1,7],],
            'b_closed_eyes':[[0,1,2,13,22],[0,1,2,12,13,23],[0,24],],
            'b_opened_eyes':[[0,1,2,3,18],[0,1,2,3,24],[0,1,3,7,24],],
        },
    },
    203:{
        'session_0':{
            'baseline':[[0,1,3,5,7,14]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    204:{
        'session_0':{
            ## TP7 highly afffected by artifacts
            ## blinking and lateral eyes movements
            # 'ica_var': 0.999,
            'baseline':[[0],[0]],
            'a_closed_eyes':[[]]*3,
            'a_opened_eyes':[[]]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },
    ################################
    ################################
    ## neuro_project_audace
    
    1:{
        'session_0':{
            'baseline':[[0],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[0,3],[0],[0],],
            'b_closed_eyes':[[2],[0],[0],],
            'b_opened_eyes':[[0],[0],[0],],
        },
        'session_1':{
            'baseline':[[0]],
            'a_closed_eyes':[[2],[],[1,12],],
            'a_opened_eyes':[[0],[0,4],[0,7,8],],
            'b_closed_eyes':[[5],[6,8],[],],
            'b_opened_eyes':[[0],[0],[0,3],],
        },
        'session_2':{
            'baseline':[[0],[0,2]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    ################
    2:{
        'session_0':{
            'baseline':[[1],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[11,],[0,],[1,2,],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
        'session_1':{
            'baseline':[[0],],
            'a_closed_eyes':[[22],[11],[],],
            'a_opened_eyes':[[0],[0],[0],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    ################
    3:{
        ## session 0 data not good
        'session_0':{
            'baseline':[[0],[]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
        ## session 1 only resting
        'session_1':{
            'baseline':[[0],],
            'a_closed_eyes':[[28],[29],[31,32],],
            'a_opened_eyes':[[1],[0],[0,33,34],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    ################
    4:{
        'session_0':{
            'baseline':[[0,6,11,15],],
            'a_closed_eyes':[[4],[1,3],[11,],],
            'a_opened_eyes':[[0,1,3],[0,1,3,4],[0,8],],
            'b_closed_eyes':[[],[0,1],[],],
            'b_opened_eyes':[[0,1,],[1],[0,2,8],],
        },
        'session_1':{
            'baseline':[[],[],[],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    ################
    5:{
        'session_0':{
            'baseline':[[0],],
            'a_closed_eyes':[[0],[0,1],[0,1],],
            'a_opened_eyes':[[0],[0],[0],],
            'b_closed_eyes':[[0],[0],[1,2],],
            'b_opened_eyes':[[0,2,5,6,8,12],[0,1,2,11,18],[0,1,2],],
        },
        'session_1':{
            'baseline':[[],[],[],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    ################
    6:{
        'session_0':{
            'baseline':[[0,1],],
            'a_closed_eyes':[[],[],[0],],
            'a_opened_eyes':[[0,14],[0,],[0,1],],
            'b_closed_eyes':[[],[],[0],],
            'b_opened_eyes':[[0,1],[0],[0],],
        },
        'session_1':{
            'baseline':[[],[],[],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    ################
    7:{
        'session_0':{
            'baseline':[[0]],
            'a_closed_eyes':[[8],[17,20,21,25],[23,24],],
            'a_opened_eyes':[[0,26],[0,19,28],[0,19,22,24],],
            'b_closed_eyes':[[0],[],[],],
            'b_opened_eyes':[[1,4],[0,17],[7],],
        },
        'session_1':{
            'baseline':[[]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },
    # 7:{
    #     'session_0':{
    #         'baseline':[[5],[],[],],
    #         'a_closed_eyes':[[],[],[],],
    #         'a_opened_eyes':[[4],[3],[3],],
    #         'b_closed_eyes':[[4],[],[],],
    #         'b_opened_eyes':[[6,13],[7],[12,14],],
    #     },
    #     'session_1':{
    #         'baseline':[[0],[],[],],
    #         'a_closed_eyes':[[],[],[19,35],],
    #         'a_opened_eyes':[[0],[0],[0],],
    #         'b_closed_eyes':[[],[],[],],
    #         'b_opened_eyes':[[0],[0],[0,13],],
    #     },
    # },
    ################
    8:{
        'session_0':{
            'baseline':[[0],],
            'a_closed_eyes':[[0,37],[],[0,1],],
            'a_opened_eyes':[[0,1,10],[0,1],[0,1],],
            'b_closed_eyes':[[0,1,41],[0,1],[0,],],
            'b_opened_eyes':[[0],[0,1,4],[0],],
        },
    }, 
    ################
    9:{
        'session_0':{
            'baseline':[[0],],
            'a_closed_eyes':[[],[],[0,34],],
            'a_opened_eyes':[[0,1],[0,1],[0,],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[0,1],[0,1],[0],],
        },
    }, 
    ################
    10:{
        'session_0':{
            'baseline':[[0,2]],
            'a_closed_eyes':[[8],[],[],],
            'a_opened_eyes':[[0],[0,1,14],[0,],],
            'b_closed_eyes':[[],[0],[],],
            'b_opened_eyes':[[0,3],[0,2],[0],],
        },
    },
    ################
    11:{
        'session_0':{
            'baseline':[[0,15],],
            'a_closed_eyes':[[2],[1],[2,10],],
            'a_opened_eyes':[[0,14,25],[0,1],[0,23],],
            'b_closed_eyes':[[1],[4],[2],],
            'b_opened_eyes':[[0],[0],[0,28],],
        },
    },
    # 11:{
    #     'session_0':{
    #         'baseline':[[0],],
    #         'a_closed_eyes':[[2],[2],[2,18],],
    #         'a_opened_eyes':[[0],[0,1],[0],],
    #         'b_closed_eyes':[[27],[1,29],[1],],
    #         'b_opened_eyes':[[0],[0],[0],],
    #     },
    # },   
    ################
    12:{
        'session_0':{
            'baseline':[[0,2,12,13],[]],
            'a_closed_eyes':[[],[23],[3,4,6],],
            'a_opened_eyes':[[30,32],[3,26,27,],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[0,15],[20,27],[25,26],],
        },
    },
    
    ## neuro_project_audace
    ################################
    ################################
}