#!/usr/bin/env python

def getKwargsDict(infile = 'ex.in'):
    """ 
    returns dictionary containing keyword arguments
    infile: text file containing keys and values separated by '='
    """
    kwargs_dict = {}
    with open(infile) as f:
        for line in f:
            if '=' in line:
                kwargs_list = line.partition('=')
                key = kwargs_list[0].strip()
                val = kwargs_list[2].strip()
    
                try:
                    val = float(val)
                except ValueError:
                    if ' ' in val:
                        val = val.split()
    
                        try:
                            val = [float(val) for val in val]
                        except ValueError:
                            pass
   
                kwargs_dict[key] = val 
    
    return kwargs_dict 

