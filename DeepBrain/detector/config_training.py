config = {'train_data_path':['/home/dwu/data/DZH_DATA/Sample3D/'],
          'val_data_path':['/home/dwu/data/DZH_DATA/Sample3D/'],
          'test_data_path':['/home/dwu/data/DZH_DATA/Sample3D/'],
          
          'train_preprocess_result_path':'/home/dwu/data/DZH_DATA/Sample3D/',
          # contains numpy for the data and label, which is generated by prepare.py
          'val_preprocess_result_path':'/home/dwu/data/DZH_DATA/Sample3D/',
          # make sure copy all the numpy into one folder after prepare.py
          'test_preprocess_result_path':'/home/dwu/data/DZH_DATA/Sample3D/',


          'black_list':[],
          
          'preprocessing_backend':'python',
         } 
