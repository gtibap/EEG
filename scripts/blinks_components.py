blinks_components_dict = {
    100:{
        'session_0':{
            'baseline':[[0]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[0],[0],[0],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[0],[0],[0],],
        },
    },
    101:{
        'session_0':{
            'baseline':[[0]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[2],[2],[2],],
            'b_closed_eyes':[[7],[],[],],
            'b_opened_eyes':[[2,6],[2],[1,8],],
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
            'ica_var': 0.999,
            'baseline':[[2,3]],
            'a_closed_eyes':[[],[11],[10],],
            'a_opened_eyes':[[],[2,9],[2,15],],
            'b_closed_eyes':[[6,27],[6,24],[16,25],],
            'b_opened_eyes':[[1,22],[1,4,30],[1,28],],
        },
    },
    1:{
        'session_0':{
            'baseline':[[3],[],[],],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[3],[4],[3],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[3],[2],[7],],
        },
        'session_1':{
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[2],[1],[2],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[1],[3],[3],],
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
            'a_closed_eyes':[[86],[],[],],
            'a_opened_eyes':[[],[],[],],
            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
    },    
}