blinks_components_dict = {
    100:{
        'session_0':{
            'baseline':[[0, 13]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[0],[0,1],[0],],
            'b_closed_eyes':[[],[],[18],],
            'b_opened_eyes':[[0,6],[0],[0,7],],
        },
    },
    101:{
        'session_0':{
            'baseline':[[0]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[1],[1],[6],],
            'b_closed_eyes':[[8],[13],[10],],
            'b_opened_eyes':[[2,4,11],[1],[1,6],],
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
            'baseline':[[2,3]],
            'a_closed_eyes':[[],[11],[10],],
            'a_opened_eyes':[[],[2,9],[2,15],],
            'b_closed_eyes':[[6,27],[6,24],[16,25],],
            'b_opened_eyes':[[1,22],[1,4,30],[1,28],],
        },
    },
    203:{
        'session_0':{
            ## TP7 highly afffected by artifacts
            ## blinking and lateral eyes movements
            # 'ica_var': 0.999,
            'baseline':[[0,6,]],
            'a_closed_eyes':[[6,24],[5,],[5,],],
            'a_opened_eyes':[[0,5,],[0,6,],[0,10,],],
            'b_closed_eyes':[[6,],[4,11,],[4,7,18,],],
            'b_opened_eyes':[[1,7,13],[1,6,14,],[1,7,14],],
        },
    },
    1:{
        'session_0':{
            'baseline':[[0],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[0,3],[0],[0],],
            'b_closed_eyes':[[2],[0],[0],],
            'b_opened_eyes':[[0],[0],[0],],
        },
        'session_1':{
            'baseline':[[],[],[],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[2],[1],[2],],
            'b_closed_eyes':[[13,24,27],[22],[],],
            'b_opened_eyes':[[1],[1,47],[1],],
        },
    },
    
    7:{
        'session_0':{
            'baseline':[[5],[],[],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[4],[3],[3],],
            'b_closed_eyes':[[4],[],[],],
            'b_opened_eyes':[[6,13],[7],[12,14],],
        },
        'session_1':{
            'baseline':[[0],[],[],],
            'a_closed_eyes':[[],[],[19,35],],
            'a_opened_eyes':[[0],[0],[0],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[0],[0],[0,13],],
        },
    },

    8:{
        'session_0':{
            'baseline':[[0],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[1],[0],[1,12,18],],
            'b_closed_eyes':[[2,3],[2],[15],],
            'b_opened_eyes':[[1,15],[1],[2],],
        },
    }, 
    10:{
        'session_0':{
            'baseline':[[],[],[],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },   
    11:{
        'session_0':{
            'baseline':[[0],],
            'a_closed_eyes':[[2],[2],[2,18],],
            'a_opened_eyes':[[0],[0,1],[0],],
            'b_closed_eyes':[[27],[1,29],[1],],
            'b_opened_eyes':[[0],[0],[0],],
        },
    },   
}