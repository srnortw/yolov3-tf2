# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:14:20 2024

@author: Serkan
"""
import zipfile

import pdb

import numpy as np
import psycopg2 as pg2

import argparse

from dotenv import load_dotenv
import os

import gdown

import pandas as pd

import tensorflow as tf
#pdb.set_trace()

#from concurrent.futures import ThreadPoolExecutor

class c_sql_d_c():
    def __init__(self,host,port,database,user,password):
        
        self.host=host
        self.port=port
        self.database=database
        self.user=user
        self.password=password
        self.conn=pg2.connect(host=self.host,port=self.port,database=self.database,user=self.user,password=self.password)
        self.cur=self.conn.cursor()

    def query(self,sqlcommand):
        # Execute the SQL command to create the table)
        self.cur.execute(sqlcommand)
        #bring
        keys=[desc[0] for desc in self.cur.description]
        li=self.cur.fetchall()
        return pd.DataFrame(li,columns=keys)



    def query_to_tfdata1(self, genf, output_signature):
        """
        Convert SQL query to tf.data.Dataset using a generator.
        output_types: tuple of tf.dtypes matching each column
        """
        dataset = tf.data.Dataset.from_generator(
            lambda: genf,
            output_signature=output_signature
            # output_signature = (
            #     tf.TensorSpec(shape=()),  # img_loc
            #     tf.TensorSpec(shape=(None, 5))  # objects list
            # )
        )
        return dataset


    def query_generator(self, sqlcommand):
        """Generator that yields one row at a time from SQL"""
        def gener():
            # with self.conn.cursor() as cur:
            #     cur.execute(sqlcommand)
            self.cur.execute(sqlcommand)
            for row in self.cur:#self.cur
                yield row
        return gener


            # yield {'img_loc': row['img_loc'],
            #  'objects':{'class_id':[o['class_id'] for o in  row['objects']],
            #             'x1':[o['x1'] for o in row['objects']],
            #             'y1':[o['y1'] for o in row['objects']],
            #             'x2':[o['x2'] for o in row['objects']],
            #             'y2':[o['y2'] for o in row['objects']]}} #row # yields tuple per row

    def query_to_tfdata(self, sqlcommand, output_signature):
        """
        Convert SQL query to tf.data.Dataset using a generator.
        output_types: tuple of tf.dtypes matching each column
        """
        dataset = tf.data.Dataset.from_generator(
            lambda: self.query_generator(sqlcommand),
            output_signature=output_signature
            # output_signature = (
            #     tf.TensorSpec(shape=()),  # img_loc
            #     tf.TensorSpec(shape=(None, 5))  # objects list
            # )
        )
        return dataset

    def process(self,sqlcommand):
        # Execute the SQL command to create the table)
        self.cur.execute(sqlcommand)
        # Commit the transaction
        self.conn.commit()

        # try:
        #     return self.cur.fetchone()[0]
        # except:
        #     return None


    def close_connection(self):
        # Close cursor and connection
        self.cur.close()
        self.conn.close()
        


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
        description='put datasets into sql database')
    
    parser.add_argument(
        '-dn',
        '--dataset_name',default='construction_safety')
    
    parser.add_argument(
        '-dsn',
        '--data_source_name',default='construction_safety')

    
    parser.add_argument(
        '-did',
        '--drive_id',
        default=',')

    parser.add_argument(
        '-cls_loc',
        '--class_file_loc',default='')

    inputs=parser.parse_args()
    
    dataset_name=inputs.dataset_name
    data_source_name=inputs.data_source_name
    drive_id=inputs.drive_id
    class_file_loc=inputs.class_file_loc

    print(class_file_loc)

    load_dotenv()
    sql_d_o = c_sql_d_c(os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_NAME'), os.getenv('DB_USER'),
                                os.getenv('DB_PASSWORD'))

    # dropts='''
    # DROP TABLE Satelimgslocs;
    # '''
    # sql_d_c.process(dropts)
    
    # Define the SQL command to create a new table

    # create_table_query = f'''
    # CREATE TABLE IF NOT EXISTS {dataset_name}(
    # id SERIAL PRIMARY KEY,
    # source VARCHAR(50),
    # loc VARCHAR(100),
    # label VARCHAR(50)
    # );
    # '''



    create_table_query=f'''CREATE TABLE IF NOT EXISTS {dataset_name}_images (
        image_id SERIAL PRIMARY KEY,
        source VARCHAR(50) NOT NULL,
        img_loc TEXT NOT NULL,
        img_resh INT,
        img_resw INT,
        img_resc INT
    );
    
    CREATE TABLE IF NOT EXISTS {dataset_name}_annotations (
    id SERIAL PRIMARY KEY,
    image_id INT REFERENCES {dataset_name}_images(image_id),
    class_id INT ,
    x1 FLOAT ,
    y1 FLOAT ,
    x2 FLOAT ,
    y2 FLOAT );
    
    
    
    
    CREATE TABLE IF NOT EXISTS {dataset_name}_class(
    id SERIAL PRIMARY KEY,
    class_id INT GENERATED BY DEFAULT AS IDENTITY (START WITH 0 INCREMENT BY 1 MINVALUE 0),
    class_name TEXT NOT NULL);
    '''



    sql_d_o.process(create_table_query)


    # os.makedirs('data', exist_ok=True)


    class_n=open(f'{class_file_loc}').read().splitlines()

    entering_classes = f'''INSERT INTO {dataset_name}_class(class_name)
    
    VALUES
    '''


    for cl in class_n:
        entering_classes += f'''('{cl}'),'''

    sql_d_o.process(entering_classes[:-1])

   
    
    zip_file_path = f"data/{data_source_name}"
    
    # Google Drive file link
    url = f"https://drive.google.com/uc?id={drive_id}"


    try:
      # Download into memory or a temporary location
      gdown.download(url,zip_file_path, quiet=False)
    except:
      print('drive_id is wrong or empty')
      
    # Open the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # List all files in the ZIP
        file_names = z.namelist()
        
        # cmnds=[]
        # for file_name in file_names:
        #     if file_name.endswith(('.png', '.jpg', '.jpeg')):
                
        #         loc_com=f'''
        #         INSERT INTO Satelimgslocs(source,loc,label)
        #         VALUES
        #         ('{data_set_name}.zip','{file_names}','{file_name.split('/')[1]}')
        #         '''
        #         cmnds.append(loc_com)

        
        # with ThreadPoolExecutor() as executor:
            
        #     executor.map(sql_d_c.process,cmnds)
            
        loc0_com=f'''
        INSERT INTO {dataset_name}_images(source,img_loc,img_resh,img_resw,img_resc)
        VALUES
        '''



        images={}
        texts={}
        for file_name in file_names:
            base=".".join(file_name.split('.')[:-1])#".".join(file_name.split('/')[-1].split('.')[:-1])

            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):

                image_array = tf.image.decode_jpeg(z.read(file_name), channels=3)#it reads from zip file

                image_shape=image_array.shape

                resh, resw, resc=image_shape[0],image_shape[1],image_shape[2]

                loc0_com += f'''('{data_source_name}','{file_name}','{resh}','{resw}','{resc}'),'''

                # loc0_com+=f'''('{data_source_name}.zip','{file_name}'),'''
                #
                # ind += 1

                images[base]=file_name
                #
                # tx.append(".".join(file_name.split('.')[:-1])+'.txt')
            elif file_name.lower().endswith('.txt'):
                texts[base]=file_name
                #image_file_name=file_name
                #label=file_name.split('/')[-2]
                #loc_com+=f'''('{data_source_name}.zip','{file_name}','{file_name.split('/')[-2]}'),'''
            # elif file_name.endswith('.txt'):
            #     tx.append(file_name)

        print(loc0_com)
        sql_d_o.process(loc0_com[:-1])

        x = f'''
        SELECT image_id,img_loc FROM {dataset_name}_images
        '''

        q=sql_d_o.query_generator(x)



        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.int32),#image_id
            tf.TensorSpec(shape=(), dtype=tf.string)# img_loc
            )


        q_r = tf.data.Dataset.from_generator(
            lambda: q(),
            output_signature=output_signature
            # output_signature = (
            #     tf.TensorSpec(shape=()),  # img_loc
            #     tf.TensorSpec(shape=(None, 5))  # objects list
            # )
        )


        #
        # q=sql_d_o.query_to_tfdata(x,output_signature)

        print(q_r)

        id_map = {
            ".".join(img_loc.numpy().decode().split('.')[:-1]): img_id.numpy()
            for img_id, img_loc in q_r
        }#os.path.splitext(os.path.basename(row["img_loc"]))[0]

        loc1_com = f'''
        INSERT INTO {dataset_name}_annotations(image_id,class_id,x1,y1,x2,y2)
        VALUES

        '''

        for base,file_name in texts.items():
            if id_map.get(base):
                with z.open(file_name) as f:
                    txt_data = f.read().decode("utf-8").strip().split("\n")

                for line in txt_data:
                    try:
                        class_id, x1, y1, x2, y2 = line.split()
                    except:
                        print(f'it is empty text or remove empty row also text loc is {file_name}')
                        break
                    loc1_com += f'''('{id_map[base]}','{class_id}','{x1}','{y1}','{x2}','{y2}'),'''

        sql_d_o.process(loc1_com[:-1])


        #
        # q=list(q['image_id'])


        # for i,j in enumerate(tx):
        #     with z.open(j) as f:
        #         txt_data = f.read().decode("utf-8").strip().split("\n")
        #
        #     for line in txt_data:
        #         line_l=line.split()
        #
        #         class_id=line_l[0]
        #         x1= line_l[1]
        #         y1 = line_l[2]
        #         x2= line_l[3]
        #         y2 = line_l[4]
        #
        #         loc1_com += f'''('{q[i]}','{class_id}','{x1}','{y1}','{x2}','{y2}'),'''
        #



        #loc_com=loc0_com[:-1]+';'+loc1_com[:-1]
        #
        # sql_d_o.process(loc1_com[:-1])

        viewcom=f'''CREATE VIEW {dataset_name}_view AS
        SELECT 
            i.image_id,i.source,i.img_loc,i.img_resh,i.img_resw,i.img_resc,
            a.class_id,a.x1, a.y1, a.x2, a.y2,
            c.class_name
        FROM {dataset_name}_images i
        JOIN {dataset_name}_annotations a
            USING(image_id)
        JOIN {dataset_name}_class c
            USING(class_id)
            ORDER BY image_id
            ;'''

        sql_d_o.process(viewcom)

        sql_d_o.close_connection()



        # p = """
        #     SELECT json_build_object(
        #         'img_loc', img_loc,
        #         'objects', json_agg(
        #             json_build_object(
        #                 'class_id', class_id,
        #                 'x1', x1,
        #                 'y1', y1,
        #                 'x2', x2,
        #                 'y2', y2
        #             )
        #         )
        #     ) AS image_data
        #     FROM animals_multilabel_view
        #     GROUP BY img_loc;
        # """
        # output_signature = (
        #     tf.TensorSpec(shape=(1,), dtype=tf.string)) # img_loc
        #
        # bbc=sql_d_o.query_to_tfdata(p, output_signature)
        #
        # for i in bbc:
        #     print(i)