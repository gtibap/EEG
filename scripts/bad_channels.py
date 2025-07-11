bad_channels_dict = {

    ##################    
    ## channels to exclude from the analysis... those channels are suceptible to artifacts
    'biosemi':[],

    'geodesic':['E38', 'E39', 'E43', 'E44', 'E45', 'E48', 'E49', 'E56', 'E57', 'E63', 'E68','E73','E81','E88','E94','E99', 'E100', 'E107', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128',],

    ###################
    100:{
        'session_0':{
            # 'general':['T8'],
            'baseline':[['FT8','T8']],
            'a_closed_eyes':[['FT8','T8'],['FT8','T8'],['FT8','T8'],],
            'a_opened_eyes':[['FT8','T8'],['F4','FT8','T8'],['FT8','T8'],],
            'b_closed_eyes':[['FT8','T8'],['FT8','T8'],['FT8','T8'],],
            'b_opened_eyes':[['FT8','T8','C6'],['FT8','T8','C6'],['FT8','T8','C6'],],
        },
    },
    101:{
        'session_0':{
            # 'general':[],
            ## POz, AF4, PO3
            'baseline':[['PO3','POz','AF4']],
            'a_closed_eyes':[['PO3','POz','AF4'],['PO3','POz','AF4'],['PO3','POz','AF4']],
            'a_opened_eyes':[['PO3','POz','AF4'],['PO3','POz','AF4'],['PO3','POz','AF4','T7']],
            'b_closed_eyes':[['PO3','POz','AF4'],['PO3','POz','AF4'],['PO3','POz','AF4']],
            'b_opened_eyes':[['PO3','POz','AF4'],['PO3','POz','AF4'],['PO3','POz','AF4']],
        },
    },
    102:{
        'session_0':{
            # 'general':[],
            'baseline':[[]],
            'a_closed_eyes':[['PO3'],['O2'],['Fpz','O2']],
            'a_opened_eyes':[[],[],['F8']],
            'b_closed_eyes':[[],[],[]],
            'b_opened_eyes':[[],[],[]],
        },
    },
    103:{
        'session_0':{
            'general':[],
            'baseline':[['PO3','POz']],
            'a_closed_eyes':[['PO3','POz'],['PO3','POz'],['PO3','POz']],
            'a_opened_eyes':[['PO3','POz'],['PO3','POz'],['PO3','POz']],
            'b_closed_eyes':[['PO3','POz'],['PO3','POz'],['PO3','POz']],
            'b_opened_eyes':[['PO3','POz'],['PO3','POz'],['PO3','POz']],
        },
    },
    104:{
        'session_0':{
            'general':[],
            ## PO3, P5, FT8, AF4
            'a_closed_eyes':[[],[],[]],
            'a_opened_eyes':[[],[],[]],
            'b_closed_eyes':[[],[],[]],
            'b_opened_eyes':[[],[],[]],
        },
    },
    105:{
        'session_0':{
            'general':[],
            'baseline':[['TP7','P7','Fp2']],
            'a_closed_eyes':[['C4','TP7','P7','Fp2'],['TP7','P7','Fp2'],['TP7','P7','Fp2']],
            'a_opened_eyes':[['C4','TP7','P7','Fp2'],['TP7','P7','Fp2'],['TP7','P7','Fp2']],
            'b_closed_eyes':[['TP7','P7','Fp2'],['TP7','P7','Fp2'],['TP7','P7','Fp2']],
            'b_opened_eyes':[['TP7','P7','Fp2'],['TP7','P7','Fp2'],['TP7','P7','Fp2']],
        },
    },
    200:{
        'session_0':{
            'general':[],
            'a_closed_eyes':[[]]*3,
            'a_opened_eyes':[[]]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },
    201:{
        'session_0':{
            'general':[],
            'a_closed_eyes':[[]]*3,
            'a_opened_eyes':[[]]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },
    202:{
        'session_0':{
            'general':[],
            'a_closed_eyes':[[]]*3,
            'a_opened_eyes':[[]]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },
    203:{
        'session_0':{
            'baseline':[['E115']],
            'a_closed_eyes':[['E115'],['E115'],['E115']],
            'a_opened_eyes':[['E115'],['E115'],['E115']],
            'b_closed_eyes':[['E115'],['E115'],['E115']],
            'b_opened_eyes':[['E115'],['E115'],['E115']],
        },
    },

    ##############################################
    ####### neuroplasticity ######################

    1:{
        'session_0':{
            'baseline':[[],[]],
            'a_closed_eyes':[[],[],[],],
            'a_opened_eyes':[[],[],[],],

            'b_closed_eyes':[[],[],[],],
            'b_opened_eyes':[[],[],[],],
        },
        'session_1':{
            'baseline':[['E106']],
            'a_closed_eyes':[['E106']]*3,
            'a_opened_eyes':[['E106']]*3,
            'b_closed_eyes':[['E106']]*3,
            'b_opened_eyes':[['E106']]*3,
        }
    },

    #################################################
    2:{
        'session_0':{
            # 'general':['E8','E55'],
            'baseline':[['E7','E55']],
            'a_closed_eyes':[['E7','E55']]*3,
            'a_opened_eyes':[['E7','E55']]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
        'session_1':{
            'baseline':[['E7','E31','E55','E62','E80','E106']],
            'a_closed_eyes':[['E7','E31','E55','E62','E80','E106']]*3,
            'a_opened_eyes':[['E7','E31','E55','E62','E80','E106']]*3,
            'b_closed_eyes':[['E7','E31','E55','E62','E80','E106']]*3,
            'b_opened_eyes':[['E7','E31','E55','E62','E80','E106']]*3,
        }
    },

    #################################################
    3:{
        'session_0':{
            # 'general':['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95'],
            ## signals degradation increases with time
            # 'a_closed_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
            # 'a_opened_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
            # 'b_closed_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
            # 'b_opened_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
            'baseline':[['E62','E85']],
            'a_closed_eyes':[['E62','E78','E85']]*3,
            'a_opened_eyes':[['E62','E78','E85']]*3,
            'b_closed_eyes':[['E55','E62','E78','E85']]*3,
            'b_opened_eyes':[['E55','E62','E78','E85']]*3,
        },
        'session_1':{
            'baseline':[['E55','E62','E106']],
            'a_closed_eyes':[['E55','E62','E106']]*3,
            'a_opened_eyes':[['E55','E62','E106']]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },

    #################################################
    4:{
        'session_0':{
            'baseline':[['E20','E54','E55','E61','E62','E67','E72','E78','E79']],
            'a_closed_eyes':[['E20','E54','E55','E61','E62','E67','E72','E78','E79']]*3,
            'a_opened_eyes':[['E20','E54','E55','E61','E62','E67','E72','E78','E79','E8','E9','E14','E25','E26',],['E20','E54','E55','E61','E62','E67','E72','E78','E79','E117'],['E20','E54','E55','E61','E62','E67','E72','E78','E79']],
            'b_closed_eyes':[['E20','E54','E55','E61','E62','E67','E72','E78','E79']]*3,
            'b_opened_eyes':[['E20','E54','E55','E61','E62','E67','E72','E78','E79']]*3,
        },
        'session_1':{
            'baseline':[['E4','E11','E16','E18','E20','E24','E28','E29','E34','E35','E50','E51','E58','E59','E64','E66','E70','E74','E75','E76','E83','E84','E90','E96','E97','E98','E101','E104','E109','E116',]],
            'a_closed_eyes':[['E4','E11','E16','E18','E20','E24','E28','E29','E34','E35','E50','E51','E58','E59','E64','E65','E66','E70','E74','E75','E76','E83','E84','E90','E96','E97','E98','E101','E104','E109','E116',]]*3,
            'a_opened_eyes':[['E4','E11','E16','E18','E20','E24','E28','E29','E34','E35','E50','E51','E58','E59','E64','E65','E66','E70','E74','E75','E76','E83','E84','E90','E96','E97','E98','E101','E104','E109','E116',]]*3,
            'b_closed_eyes':[['E4','E11','E16','E18','E20','E24','E28','E29','E34','E35','E50','E51','E58','E59','E64','E65','E66','E70','E74','E75','E76','E83','E84','E90','E96','E97','E98','E101','E104','E109','E116',]]*3,
            'b_opened_eyes':[['E4','E11','E16','E18','E20','E24','E28','E29','E34','E35','E50','E51','E58','E59','E64','E65','E66','E70','E74','E75','E76','E83','E84','E90','E96','E97','E98','E101','E104','E109','E116',]]*3,
        }
    },

    #################################################
    5:{
        'session_0':{
            'baseline':[['E20',]],
            'a_closed_eyes':[['E20',]]*3,
            'a_opened_eyes':[['E8','E20',]]*3,
            'b_closed_eyes':[['E20',]]*3,
            'b_opened_eyes':[['E8','E20',]]*3,
        },
        'session_1':{
            'baseline':[['E62']],
            'a_closed_eyes':[['E62']]*3,
            'a_opened_eyes':[['E62']]*3,
            'b_closed_eyes':[['E62']]*3,
            'b_opened_eyes':[['E62']]*3,
        },

    },

    #################################################
    6:{
        'session_0':{
            'baseline':[[]],
            'a_closed_eyes':[['E37','E42','E116']]*3,
            'a_opened_eyes':[['E25','E37','E42']]*3,
            'b_closed_eyes':[['E37','E40','E42','E84','E92','E103','E116']]*3,
            'b_opened_eyes':[['E37','E40','E42','E92','E96','E103','E116']]*3,
        },
        'session_1':{
            'baseline':[[]],
            'a_closed_eyes':[[]]*3,
            'a_opened_eyes':[[]]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },

    #################################################
    7:{
        'session_0':{
            'baseline':[['E20']],
            'a_closed_eyes':[['E20']]*3,
            'a_opened_eyes':[['E20']]*3,
            'b_closed_eyes':[['E20']]*3,
            'b_opened_eyes':[['E20']]*3,
        },
        'session_1':{
            # 'general':[],
            'baseline':[[]],
            'a_closed_eyes':[[]]*3,
            'a_opened_eyes':[[]]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },

    #################################################
    8:{
        'session_0':{
            # 'general':[],
            'baseline':[['E60','E69','E70']]*3,
            'a_closed_eyes':[['E60','E69','E70']]*3,
            'a_opened_eyes':[['E60','E69','E70']]*3,
            'b_closed_eyes':[['E60','E69','E70']]*3,
            'b_opened_eyes':[['E60','E69','E70']]*3,
        },
    },
    # 8:{
    #     'session_0':{
    #         # 'general':[],
    #         'baseline':[['E48']],
    #         'a_closed_eyes':[['E48'],['E48',],['E48','E69']],
    #         'a_opened_eyes':[['E48'],['E48'],['E48','E69']],
    #         'b_closed_eyes':[['E48','E56'],['E48'],['E48']],
    #         'b_opened_eyes':[['E48'],['E48'],['E48']],
    #     },
    # },

    #################################################
    9:{
        'session_0':{
            'baseline':[['E31','E69','E80']],
            'a_closed_eyes':[['E31','E69','E80']]*3,
            'a_opened_eyes':[['E31','E69','E80']]*3,
            'b_closed_eyes':[['E31','E69','E80']]*3,
            'b_opened_eyes':[['E31','E69','E80']]*3,
        },
    },

    #################################################
    10:{
        'session_0':{
            ## In the list 'general' we define channels that are usually affected by artefacts due their location, i.e. channels close to the ears and the eyes. Those selected channels will be excluded from the begining of signals preprocessing
            'baseline':[['E62','E72']],
            'a_closed_eyes':[['E62','E72']]*3,
            'a_opened_eyes':[['E62','E72']]*3,
            'b_closed_eyes':[['E62','E72']]*3,
            'b_opened_eyes':[['E62','E72']]*3,
            # 'general':['E38', 'E39', 'E43', 'E44', 'E45', 'E48', 'E49', 'E56', 'E57', 'E63', 'E99', 'E100', 'E107', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128', ],
            # 'baseline':[['E44','E62','E72','E107','E119','E120','E121','E125',]],
            # 'a_closed_eyes':[['E44','E62','E72','E107','E119','E120','E121','E125',]]*3,
            # 'a_opened_eyes':[['E44','E62','E72','E107','E119','E120','E121','E125',]]*3,
            # 'b_closed_eyes':[['E44','E62','E72','E107','E119','E120','E121','E125',]]*3,
            # 'b_opened_eyes':[['E44','E62','E72','E107','E119','E120','E121','E125',]]*3,
        },
    },

    #################################################
    11:{
        # 'general':['E38', 'E39', 'E43', 'E44', 'E45', 'E48', 'E49', 'E56', 'E57', 'E63', 'E68','E73','E81','E88','E94','E99', 'E100', 'E107', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128',],

        'session_0':{
            # 'general':['E38', 'E39', 'E43', 'E44', 'E45', 'E48', 'E49', 'E56', 'E57', 'E63', 'E68','E73','E81','E88','E94','E99', 'E100', 'E107', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128',],
            'baseline':[['E34','E116']],
            'a_closed_eyes':[['E34','E116']]*3,
            'a_opened_eyes':[['E34','E116']]*3,
            'b_closed_eyes':[['E34','E116']]*3,
            'b_opened_eyes':[['E34','E116']]*3,
        },
    },

    #################################################
    12:{
        # 'general':['E38', 'E39', 'E43', 'E44', 'E45', 'E48', 'E49', 'E56', 'E57', 'E63', 'E68','E73','E81','E88','E94','E99', 'E100', 'E107', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128',],

        'session_0':{
            # 'general':['E38', 'E39', 'E43', 'E44', 'E45', 'E48', 'E49', 'E56', 'E57', 'E63', 'E68','E73','E81','E88','E94','E99', 'E100', 'E107', 'E108', 'E113', 'E114', 'E115', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128',],
            'baseline':[['E7','E31','E80','E106']]*2,
            'a_closed_eyes':[['E7','E31','E80','E106']]*3,
            'a_opened_eyes':[['E7','E31','E80','E106'],['E7','E31','E54','E80','E98','E106'],['E7','E25','E31','E80','E98','E106']],
            'b_closed_eyes':[['E7','E31','E80','E98','E106']]*3,
            'b_opened_eyes':[['E7','E31','E80','E98','E106']]*3,
        },
    },

    #################################################
    'ref':{
        'session_0':{
            'general':[],
            'baseline':[[]],
            'a_closed_eyes':[[]]*3,
            'a_opened_eyes':[[]]*3,
            'b_closed_eyes':[[]]*3,
            'b_opened_eyes':[[]]*3,
        },
    },
}