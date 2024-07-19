from PIL import Image
import numpy as np
import random
import math
from datetime import datetime
import time
import pandas as pd 

def image(file):
    img = Image.open(file)
    k = np.array(img)
    k1=np.array(img)
    return k,k1

def inp(file1):
    f=open(file1,"r")
    m=f.read()
    #print(len(m))
    return m

def ran_key():
    k = random.randint(2,255)
    #print(k)
    return k

def header_prep(length_message, key):
    eight="01111110"+format(length_message,"016b") + format(key,'08b')
    return eight

def encryption(m,key):
    enc=[]
    for i in m:
        nu=ord(i)
        #print(i, nu)
        enc.append(nu^key)
    return enc

def binary_con(s):
    num = ""
    for i in s:
        num += format(i, "08b")
    return num

def rev_convert(s):
    return int("".join(str(i) for i in s), 2)

def embedd(array, num):   
    l = 0
    l1 = len(num)
    i = 0
    while i < len(array) and l < l1:
        j = 0
        while j < len(array[0]) and l < l1:
            k = 0
            while k < 3 and l < l1:
                r = array[i][j][k] % 2
                if r == 0 and num[l] == '1':
                    array[i][j][k] += 1
                elif r == 1 and num[l] == '0':
                    array[i][j][k] -= 1
                k += 1
                l += 1
            j += 1
        i += 1
    
def extract_header(array, header_size):
    l = 0
    i = 0
    msg = ""
    while i < len(array) and l < header_size :
        j = 0
        while j < len(array[0]) and l < header_size:
            k = 0
            while k < 3 and l < header_size:
                r = array[i][j][k] % 2
                msg += str(r)
                k += 1
                l += 1
            j += 1
        i += 1
    return msg

def merge(exract1):
    new =[]
    for i in range(0, len(extract1), 8):
        b = extract1[i:i+8]
        new.append(rev_convert(b))
    return new

def extract(array, l1):
    l = 0
    i = 0
    msg = ""
    while i < len(array) and l < l1:
        j = 0
        while j < len(array[0]) and l < l1:
            k = 0
            while k < 3 and l < l1:
                r = array[i][j][k] % 2
                msg += str(r)
                k += 1
                l += 1
            j += 1
        i += 1
    return msg

def decryption(enc, key):
    
    re=''
    for i in enc:
        d = i ^ key
        re=re+chr(d)
    return re


def rme(array1, array):
    s=0
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                diff = int(array1[i][j][k]) - int(array[i][j][k])
                ss = diff*diff
                s=s+diff
    return  s / (height*width*3)

def rmse(MSE):
    root_mean=MSE**0.5
    return root_mean

def psnr(sizemax , mse):
    if(mse==0.0):
        mse=0.00001
    dupsnr = sizemax ** 2 / mse
    PSNR = 10*math.log(dupsnr,10)
    return PSNR

def ssim(array, array1):
    mux=np.array(array)
    xavg=np.mean(mux)
    muy=np.array(array1)
    yavg=np.mean(muy)    
    stdx=np.std(array)
    stdy=np.std(array1)   
    varx=np.var(array)
    vary=np.var(array1)
    
    lumm = (2*xavg*yavg) / (xavg**2 + yavg**2)
    contrast = (2*stdx*stdy ) / (stdx**2 + stdy**2)
    result_value = lumm * contrast;
    return result_value

def ae(array, array1):
    s=0
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                diff = abs( int(array1[i][j][k]) - int(array[i][j][k]))
                s += diff            
    return  s / (height*width*3)
 
def ad(array, array1):
    s=0
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                diff =  int(array1[i][j][k]) - int(array[i][j][k])
                s += diff
    s= s / (height*width*3)
    return  s

def ncc(array,array1):
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                dn = int(array[i][j][k]) * int(array[i][j][k])
    #print("NCC down",dn)

    for i in range(width):
        for j in range(height):
            for k in range(3):
                up = int(array[i][j][k]) * int(array1[i][j][k])
    #print("NCC up",up)
                ans = up / dn
    return ans

def imf(array,array1):
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                diff = int(array1[i][j][k]) - int(array[i][j][k])
                upper = diff*diff

    for i in range(width):
        for j in range(height):
            for k in range(3):
                lower = int(array[i][j][k]) * int(array[i][j][k])

    ress = upper / lower
    ress1 = 1 - (ress)
    return ress1

def sc(array,array1):
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                dnn = int(array1[i][j][k]) * int(array1[i][j][k])

    for i in range(width):
        for j in range(height):
            for k in range(3):
                upp = int(array[i][j][k]) * int(array[i][j][k])

    anss = upp / dnn
    return anss
    
def work(image1,file1, result):                        
    header_size= 32
    print("OPERATIONS\n1.EMBEDDING\n2.EXTRACTION")
    ch=int(input("ENTER OPERATION:"))
    print(image1)
    ind_result=[]
    ind_result.append(image1)
    if(ch==1):
        s=inp(file1)
        key = ran_key()
        startenc = datetime.now()
        enc = encryption(s,key)
        endenc = datetime.now()
        tmeenc = endenc - startenc
    
    
        head=header_prep(len(s), key)
        num=head+binary_con(enc)
        embsize = len(num)

        array1, array = image(image1)
        width = len(array) ; height = len(array[0]); total = width * height*3
        sizemax=np.max(array1)
        #print("IMAGE WIDTH = ",len(array))
        #print("IMAGE HEIGHT = ",len(array[0]))
        print("TOTAL EMBEDDING CAPACITY             = ",total)
        if(total>=embsize):
            start = datetime.now()
            emit = embedd(array, num)
            end = datetime.now() 
            immm = Image.fromarray(array)
            immm.save('vasanthganesh.tif')
            emit = (end-start)
            print()
            print("QUALITY ANALYSIS PARAMETERS")
            #print("``````` ```````` ``````````")
            MSE = rme(array1, array)
            print("MEAN SQUARE ERROR(mse)                      : ",MSE)
            RMSE = rmse(MSE)
            print("ROOT MEAN SQUARE ERROR(rmse)                : ",RMSE)
            PSNR = psnr(sizemax , MSE)
            print("PEAK SIGNAL TO NOISE RATIO(psnr)            : ",PSNR)
            SSIM = ssim(array, array1)
            print("STRUCTURAL SIMILARITY INDEX MATRIX(ssim)    : ",SSIM)
            AE = ae(array, array1)
            print("ABSOLUTE ERROR(ae)                          : ",AE)
            AD = ad(array, array1)
            print("ABSOLUTE DIFFERENCE(ad)                     : ",AD)
            NCC = ncc(array, array1)
            print("STANDARD NORMALIZED CROSS-CORRELATION(ncc)  : ",NCC)
            IF = imf(array, array1)
            print("IMAGE FIDELITY(if)                          : ",IF)
            SC = sc(array, array1)
            print("STRUCTURAL CONTENT(sc)                      : ",SC)

            print("EMBEDDING TIME IN BITS                      : ",emit.total_seconds(),"milliseconds")
            print("ENCRYPTION TIME                             : ",tmeenc.total_seconds(),"milliseconds")
            ind_result.append(MSE)
            ind_result.append(RMSE)
            ind_result.append(PSNR)
            ind_result.append(SSIM)
            ind_result.append(AE)
            ind_result.append(AD)
            ind_result.append(NCC)
            ind_result.append(IF)
            ind_result.append(SC)
            ind_result.append(emit.total_seconds())
            ind_result.append(tmeenc.total_seconds())
            result.append(ind_result)

        else:
            print("NOPE CANNOT BE EMBEDEDD, PLEASE SELECT NEW IMAGE")
    
    
    elif(ch==2):
        inpp= input("ENTER IMAGE NAME:")
        array1, array = image(inpp)
        msg = extract_header(array, header_size)
        form = rev_convert (msg[:8])
        leng = rev_convert (msg [ 8:24 ])
        key1 = rev_convert (msg[24:])

        if(form ==126):
        
            extract1 = extract(array, leng*8+header_size)      
            extract1 = extract1[header_size:]
            new=merge(extract1)
            start = datetime.now()
            re = decryption(new,key1)
            end = datetime.now()
            dectme = end - start
            f=open("yasin.txt","w",encoding = "utf-8")
            f.write(re)
            f.close()
            print("DECRYPTION TIME  =  ", dectme.total_seconds(),"milliseconds")
        else:
            print("DUE TO INSUFFICIENT DATA, MESSAGE CANNOT BE EXTRACTED....")
            print("TRY USING ANOTHER IMAGE")

image_details = ["obito.jpg","mikey.jpg"]
file_details = ["test.txt"]
result=[]
for i in image_details:
    work(i,"test.txt", result)
print(result)
arr = np.asarray(result)
pd.DataFrame(arr).to_csv('samplettt.csv')  


