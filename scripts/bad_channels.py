bad_channels_dict = {
    100:{
        'general':['T8'],
        # 'a_closed_eyes':[['F4','FT8','T8'],['FT8','T8'],['FT8','T8'],['FT8','T8'],],
        # 'a_opened_eyes':[['F4','FT8','T8'],['FT8','T8'],['FT8','T8'],['FT8','T8'],],
        'a_closed_eyes':[[],[],[],[]],
        'a_opened_eyes':[[],[],[],[]],
    
        # 'b_closed_eyes':[['FT8',],['FT8',],['FT8',],],
        # 'b_opened_eyes':[['FT8','T8'],['FT8','T8'],['FT8','T8'],],
        'b_closed_eyes':[[],[],[]],
        'b_opened_eyes':[[],[],[]],

        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[],[],[],[]],
        'a_opened_eyes_excl':[[],[],[],[]],
        'b_closed_eyes_excl':[[],[],[]],
        'b_opened_eyes_excl':[[],[],[]],
    },
    101:{
        ## POz, AF4, PO3
        'a_closed_eyes':[[],[],[]],
        'a_opened_eyes':[[],[],[]],
        'b_closed_eyes':[[],[],[]],
        'b_opened_eyes':[[],[],[]],

        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[],[],[]],
        'a_opened_eyes_excl':[[],[],[]],
        'b_closed_eyes_excl':[[],[],[]],
        'b_opened_eyes_excl':[[],[],[]],
    },
    102:{
        'a_closed_eyes':[[],[],[]],
        'a_opened_eyes':[[],[],[]],
        'b_closed_eyes':[[],[],[]],
        'b_opened_eyes':[[],[],[]],
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[],[],[]],
        'a_opened_eyes_excl':[[],[],[]],
        'b_closed_eyes_excl':[[],[],[]],
        'b_opened_eyes_excl':[[],[],[]],
    },
    103:{
        'a_closed_eyes':[[],[],[]],
        'a_opened_eyes':[[],[],[]],
        'b_closed_eyes':[[],[],[]],
        'b_opened_eyes':[[],[],[]],
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[],[],[]],
        'a_opened_eyes_excl':[[],[],[]],
        'b_closed_eyes_excl':[[],[],[]],
        'b_opened_eyes_excl':[[],[],[]],
    },
    104:{
        ## PO3, P5, FT8, AF4
        'a_closed_eyes':[[],[],[]],
        'a_opened_eyes':[[],[],[]],
        'b_closed_eyes':[[],[],[]],
        'b_opened_eyes':[[],[],[]],
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[],[],[]],
        'a_opened_eyes_excl':[[],[],[]],
        'b_closed_eyes_excl':[[],[],[]],
        'b_opened_eyes_excl':[[],[],[]],
    },
    105:{
        'a_closed_eyes':[[],[],[]],
        'a_opened_eyes':[[],[],[]],
        'b_closed_eyes':[[],[],[]],
        'b_opened_eyes':[[],[],[]],
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[],[],[]],
        'a_opened_eyes_excl':[[],[],[]],
        'b_closed_eyes_excl':[[],[],[]],
        'b_opened_eyes_excl':[[],[],[]],
    },
    200:{
        'a_closed_eyes':[[]]*3,
        'a_opened_eyes':[[]]*3,
        'b_closed_eyes':[[]]*3,
        'b_opened_eyes':[[]]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[]]*3,
        'a_opened_eyes_excl':[[]]*3,
        'b_closed_eyes_excl':[[]]*3,
        'b_opened_eyes_excl':[[]]*3,
    },
    201:{
        'a_closed_eyes':[[]]*3,
        'a_opened_eyes':[[]]*3,
        'b_closed_eyes':[[]]*3,
        'b_opened_eyes':[[]]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[]]*3,
        'a_opened_eyes_excl':[[]]*3,
        'b_closed_eyes_excl':[[]]*3,
        'b_opened_eyes_excl':[[]]*3,
    },
    202:{
        'a_closed_eyes':[[]]*3,
        'a_opened_eyes':[[]]*3,
        'b_closed_eyes':[[]]*3,
        'b_opened_eyes':[[]]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[]]*3,
        'a_opened_eyes_excl':[[]]*3,
        'b_closed_eyes_excl':[[]]*3,
        'b_opened_eyes_excl':[[]]*3,
    },
    1:{
        'general':['E48','E119'],
        'a_closed_eyes':[['E48','E119']]*3,
        'a_opened_eyes':[['E48','E119']]*3,
        'b_closed_eyes':[['E48','E119']]*3,
        'b_opened_eyes':[['E48','E119']]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[['E48','E119']]*3,
        'a_opened_eyes_excl':[['E48','E119']]*3,
        'b_closed_eyes_excl':[['E48','E119']]*3,
        'b_opened_eyes_excl':[['E48','E119']]*3,
    },
    2:{
        'general':['E8','E55'],
        'a_closed_eyes':[['E8','E55']]*3,
        'a_opened_eyes':[['E8','E55']]*3,
        'b_closed_eyes':[['E8','E55']]*3,
        'b_opened_eyes':[['E8','E55']]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[['E8']]*3,
        'a_opened_eyes_excl':[['E8']]*3,
        'b_closed_eyes_excl':[['E8']]*3,
        'b_opened_eyes_excl':[['E8']]*3,
    },
    3:{
        ## signals degradation increases with time
        'a_closed_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
        'a_opened_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
        'b_closed_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
        'b_opened_eyes':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
        'a_opened_eyes_excl':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
        'b_closed_eyes_excl':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
        'b_opened_eyes_excl':[['E45','E49','E56','E57','E62','E72','E78','E79','E85','E86','E92','E95']]*3,
    },
    4:{
        'a_closed_eyes':[['E20','E44','E48','E114']]*3,
        'a_opened_eyes':[['E20','E44','E48','E114']]*3,
        'b_closed_eyes':[['E20','E44','E48','E114']]*3,
        'b_opened_eyes':[['E20','E44','E48','E114']]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[['E44','E48','E114']]*3,
        'a_opened_eyes_excl':[['E44','E48','E114']]*3,
        'b_closed_eyes_excl':[['E44','E48','E114']]*3,
        'b_opened_eyes_excl':[['E44','E48','E114']]*3,
    },
    5:{
        'a_closed_eyes':[['E20','E43','E44','E113']]*3,
        'a_opened_eyes':[['E20','E43','E44','E113']]*3,
        'b_closed_eyes':[['E20','E43','E44','E113']]*3,
        'b_opened_eyes':[['E20','E43','E44','E113']]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[['E43','E44','E113']]*3,
        'a_opened_eyes_excl':[['E43','E44','E113']]*3,
        'b_closed_eyes_excl':[['E43','E44','E113']]*3,
        'b_opened_eyes_excl':[['E43','E44','E113']]*3,

    },
    6:{
        'a_closed_eyes':[['E37','E43','E48']]*3,
        'a_opened_eyes':[['E37','E43','E48']]*3,
        'b_closed_eyes':[['E37','E43','E48']]*3,
        'b_opened_eyes':[['E37','E43','E48']]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[['E43','E48']]*3,
        'a_opened_eyes_excl':[['E43','E48']]*3,
        'b_closed_eyes_excl':[['E43','E48']]*3,
        'b_opened_eyes_excl':[['E43','E48']]*3,

    },
    7:{
        'general':[],
        'a_closed_eyes':[['E48']]*3,
        'a_opened_eyes':[['E48']]*3,
        'b_closed_eyes':[['E48']]*3,
        'b_opened_eyes':[['E48']]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[['E48']]*3,
        'a_opened_eyes_excl':[['E48']]*3,
        'b_closed_eyes_excl':[['E48']]*3,
        'b_opened_eyes_excl':[['E48']]*3,
    },
    'ref':{
        'a_closed_eyes':[[]]*3,
        'a_opened_eyes':[[]]*3,
        'b_closed_eyes':[[]]*3,
        'b_opened_eyes':[[]]*3,
        ## channels to exclude from interpolation because close to the head net borders
        'a_closed_eyes_excl':[[]]*3,
        'a_opened_eyes_excl':[[]]*3,
        'b_closed_eyes_excl':[[]]*3,
        'b_opened_eyes_excl':[[]]*3,
    },
}