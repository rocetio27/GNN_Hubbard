import torch
def create_and_save_data(U, t):
    NCn=20
    Size = NCn*NCn

    sing_s1 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s1[0]=1
    sing_s1[1]=1
    sing_s1[2]=1
    sing_s1[3]=0
    sing_s1[4]=1
    sing_s1[5]=1
    sing_s1[6]=0
    sing_s1[7]=1
    sing_s1[8]=0
    sing_s1[9]=0
    sing_s1[10]=1
    sing_s1[11]=1
    sing_s1[12]=0
    sing_s1[13]=1
    sing_s1[14]=0
    sing_s1[15]=0
    sing_s1[16]=1
    sing_s1[17]=0
    sing_s1[18]=0
    sing_s1[19]=0

    sing_s2 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s2[0]=1
    sing_s2[1]=1
    sing_s2[2]=0
    sing_s2[3]=1
    sing_s2[4]=1
    sing_s2[5]=0
    sing_s2[6]=1
    sing_s2[7]=0
    sing_s2[8]=1
    sing_s2[9]=0
    sing_s2[10]=1
    sing_s2[11]=0
    sing_s2[12]=1
    sing_s2[13]=0
    sing_s2[14]=1
    sing_s2[15]=0
    sing_s2[16]=0
    sing_s2[17]=1
    sing_s2[18]=0
    sing_s2[19]=0

    sing_s3 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s3[0]=1
    sing_s3[1]=0
    sing_s3[2]=1
    sing_s3[3]=1
    sing_s3[4]=0
    sing_s3[5]=1
    sing_s3[6]=1
    sing_s3[7]=0
    sing_s3[8]=0
    sing_s3[9]=1
    sing_s3[10]=0
    sing_s3[11]=1
    sing_s3[12]=1
    sing_s3[13]=0
    sing_s3[14]=0
    sing_s3[15]=1
    sing_s3[16]=0
    sing_s3[17]=0
    sing_s3[18]=1
    sing_s3[19]=0

    sing_s4 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s4[0]=0
    sing_s4[1]=1
    sing_s4[2]=1
    sing_s4[3]=1
    sing_s4[4]=0
    sing_s4[5]=0
    sing_s4[6]=0
    sing_s4[7]=1
    sing_s4[8]=1
    sing_s4[9]=1
    sing_s4[10]=0
    sing_s4[11]=0
    sing_s4[12]=0
    sing_s4[13]=1
    sing_s4[14]=1
    sing_s4[15]=1
    sing_s4[16]=0
    sing_s4[17]=0
    sing_s4[18]=0
    sing_s4[19]=1

    sing_s5 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s5[0]=0
    sing_s5[1]=0
    sing_s5[2]=0
    sing_s5[3]=0
    sing_s5[4]=1
    sing_s5[5]=1
    sing_s5[6]=1
    sing_s5[7]=1
    sing_s5[8]=1
    sing_s5[9]=1
    sing_s5[10]=0
    sing_s5[11]=0
    sing_s5[12]=0
    sing_s5[13]=0
    sing_s5[14]=0
    sing_s5[15]=0
    sing_s5[16]=1
    sing_s5[17]=1
    sing_s5[18]=1
    sing_s5[19]=1

    sing_s6 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s6[0]=0
    sing_s6[1]=0
    sing_s6[2]=0
    sing_s6[3]=0
    sing_s6[4]=0
    sing_s6[5]=0
    sing_s6[6]=0
    sing_s6[7]=0
    sing_s6[8]=0
    sing_s6[9]=0
    sing_s6[10]=1
    sing_s6[11]=1
    sing_s6[12]=1
    sing_s6[13]=1
    sing_s6[14]=1
    sing_s6[15]=1
    sing_s6[16]=1
    sing_s6[17]=1
    sing_s6[18]=1
    sing_s6[19]=1

    # 여기까지 각 사이트의 각 스핀의 전자수를 함 써봣다. ㅇㅋ 여기까지 처리함 #############

    H_list = [[0 for w in range(Size)] for h in range(Size)] #0으로 다 초기화 합시다.

    for la in range(Size):
      H_list[la][la]=U # 일단 모든 H의 대각 요소를 U로 초기화 해준다.

    for la2 in range(NCn):
      H_list[(NCn+1)*la2][(NCn+1)*la2]=3*U  # 그 중에서 NCn개는 3*U이다.

    for la3 in range(NCn):
      H_list[(NCn-1)*(la3+1)][(NCn-1)*(la3+1)]=0   # 그 중에서 NCn개는 0이다.

    # 여기까지 U에 관한 해밀토니안 다 만든 것이 아니라 2*U를 해줘야 한다!!!!!
    # 이것은 순수 노가다이다. ㅋ

    for lau1 in range(1):          #  이 for문은 반복이 한번만 된다 즉 반복의 의미가 없는데 그냥 내가 코드 보기 쉽게 넣은거임.
      H_list[1][1]=2*U
      H_list[2][2]=2*U
      H_list[3][3]=2*U
      H_list[4][4]=2*U
      H_list[5][5]=2*U
      H_list[6][6]=2*U
      H_list[10][10]=2*U
      H_list[11][11]=2*U
      H_list[12][12]=2*U           ##############################################
      H_list[NCn*1 + 0][NCn*1 + 0]=2*U
      H_list[NCn*1 + 2][NCn*1 + 2]=2*U
      H_list[NCn*1 + 3][NCn*1 + 3]=2*U
      H_list[NCn*1 + 4][NCn*1 + 4]=2*U
      H_list[NCn*1 + 7][NCn*1 + 7]=2*U
      H_list[NCn*1 + 8][NCn*1 + 8]=2*U
      H_list[NCn*1 + 10][NCn*1 + 10]=2*U
      H_list[NCn*1 + 13][NCn*1 + 13]=2*U
      H_list[NCn*1 + 14][NCn*1 + 14]=2*U           ##############################################
      H_list[NCn*2 + 0][NCn*2 + 0]=2*U
      H_list[NCn*2 + 1][NCn*2 + 1]=2*U
      H_list[NCn*2 + 3][NCn*2 + 3]=2*U
      H_list[NCn*2 + 5][NCn*2 + 5]=2*U
      H_list[NCn*2 + 7][NCn*2 + 7]=2*U
      H_list[NCn*2 + 9][NCn*2 + 9]=2*U
      H_list[NCn*2 + 11][NCn*2 + 11]=2*U
      H_list[NCn*2 + 13][NCn*2 + 13]=2*U
      H_list[NCn*2 + 15][NCn*2 + 15]=2*U           ##############################################
      H_list[NCn*3 + 0][NCn*3 + 0]=2*U
      H_list[NCn*3 + 1][NCn*3 + 1]=2*U
      H_list[NCn*3 + 2][NCn*3 + 2]=2*U
      H_list[NCn*3 + 6][NCn*3 + 6]=2*U
      H_list[NCn*3 + 8][NCn*3 + 8]=2*U
      H_list[NCn*3 + 9][NCn*3 + 9]=2*U
      H_list[NCn*3 + 12][NCn*3 + 12]=2*U
      H_list[NCn*3 + 14][NCn*3 + 14]=2*U
      H_list[NCn*3 + 15][NCn*3 + 15]=2*U           ##############################################
      H_list[NCn*4 + 0][NCn*4 + 0]=2*U
      H_list[NCn*4 + 1][NCn*4 + 1]=2*U
      H_list[NCn*4 + 5][NCn*4 + 5]=2*U
      H_list[NCn*4 + 6][NCn*4 + 6]=2*U
      H_list[NCn*4 + 7][NCn*4 + 7]=2*U
      H_list[NCn*4 + 8][NCn*4 + 8]=2*U
      H_list[NCn*4 + 10][NCn*4 + 10]=2*U
      H_list[NCn*4 + 16][NCn*4 + 16]=2*U
      H_list[NCn*4 + 17][NCn*4 + 17]=2*U           ##############################################
      H_list[NCn*5 + 0][NCn*5 + 0]=2*U
      H_list[NCn*5 + 2][NCn*5 + 2]=2*U
      H_list[NCn*5 + 4][NCn*5 + 4]=2*U
      H_list[NCn*5 + 6][NCn*5 + 6]=2*U
      H_list[NCn*5 + 7][NCn*5 + 7]=2*U
      H_list[NCn*5 + 9][NCn*5 + 9]=2*U
      H_list[NCn*5 + 11][NCn*5 + 11]=2*U
      H_list[NCn*5 + 16][NCn*5 + 16]=2*U
      H_list[NCn*5 + 18][NCn*5 + 18]=2*U           ##############################################
      H_list[NCn*6 + 0][NCn*6 + 0]=2*U
      H_list[NCn*6 + 3][NCn*6 + 3]=2*U
      H_list[NCn*6 + 4][NCn*6 + 4]=2*U
      H_list[NCn*6 + 5][NCn*6 + 5]=2*U
      H_list[NCn*6 + 8][NCn*6 + 8]=2*U
      H_list[NCn*6 + 9][NCn*6 + 9]=2*U
      H_list[NCn*6 + 12][NCn*6 + 12]=2*U
      H_list[NCn*6 + 17][NCn*6 + 17]=2*U
      H_list[NCn*6 + 18][NCn*6 + 18]=2*U           ##############################################
      H_list[NCn*7 + 1][NCn*7 + 1]=2*U
      H_list[NCn*7 + 2][NCn*7 + 2]=2*U
      H_list[NCn*7 + 4][NCn*7 + 4]=2*U
      H_list[NCn*7 + 5][NCn*7 + 5]=2*U
      H_list[NCn*7 + 8][NCn*7 + 8]=2*U
      H_list[NCn*7 + 9][NCn*7 + 9]=2*U
      H_list[NCn*7 + 13][NCn*7 + 13]=2*U
      H_list[NCn*7 + 16][NCn*7 + 16]=2*U
      H_list[NCn*7 + 19][NCn*7 + 19]=2*U           ##############################################
      H_list[NCn*8 + 1][NCn*8 + 1]=2*U
      H_list[NCn*8 + 3][NCn*8 + 3]=2*U
      H_list[NCn*8 + 4][NCn*8 + 4]=2*U
      H_list[NCn*8 + 6][NCn*8 + 6]=2*U
      H_list[NCn*8 + 7][NCn*8 + 7]=2*U
      H_list[NCn*8 + 9][NCn*8 + 9]=2*U
      H_list[NCn*8 + 14][NCn*8 + 14]=2*U
      H_list[NCn*8 + 17][NCn*8 + 17]=2*U
      H_list[NCn*8 + 19][NCn*8 + 19]=2*U           ##############################################
      H_list[NCn*9 + 2][NCn*9 + 2]=2*U
      H_list[NCn*9 + 3][NCn*9 + 3]=2*U
      H_list[NCn*9 + 5][NCn*9 + 5]=2*U
      H_list[NCn*9 + 6][NCn*9 + 6]=2*U
      H_list[NCn*9 + 7][NCn*9 + 7]=2*U
      H_list[NCn*9 + 8][NCn*9 + 8]=2*U
      H_list[NCn*9 + 15][NCn*9 + 15]=2*U
      H_list[NCn*9 + 18][NCn*9 + 18]=2*U
      H_list[NCn*9 + 19][NCn*9 + 19]=2*U           ##############################################
      H_list[NCn*10 + 0][NCn*10 + 0]=2*U
      H_list[NCn*10 + 1][NCn*10 + 1]=2*U
      H_list[NCn*10 + 4][NCn*10 + 4]=2*U
      H_list[NCn*10 + 11][NCn*10 + 11]=2*U
      H_list[NCn*10 + 12][NCn*10 + 12]=2*U
      H_list[NCn*10 + 13][NCn*10 + 13]=2*U
      H_list[NCn*10 + 14][NCn*10 + 14]=2*U
      H_list[NCn*10 + 16][NCn*10 + 16]=2*U
      H_list[NCn*10 + 17][NCn*10 + 17]=2*U           ##############################################
      H_list[NCn*11 + 0][NCn*11 + 0]=2*U
      H_list[NCn*11 + 2][NCn*11 + 2]=2*U
      H_list[NCn*11 + 5][NCn*11 + 5]=2*U
      H_list[NCn*11 + 10][NCn*11 + 10]=2*U
      H_list[NCn*11 + 12][NCn*11 + 12]=2*U
      H_list[NCn*11 + 13][NCn*11 + 13]=2*U
      H_list[NCn*11 + 15][NCn*11 + 15]=2*U
      H_list[NCn*11 + 16][NCn*11 + 16]=2*U
      H_list[NCn*11 + 18][NCn*11 + 18]=2*U           ##############################################
      H_list[NCn*12 + 0][NCn*12 + 0]=2*U
      H_list[NCn*12 + 3][NCn*12 + 3]=2*U
      H_list[NCn*12 + 6][NCn*12 + 6]=2*U
      H_list[NCn*12 + 10][NCn*12 + 10]=2*U
      H_list[NCn*12 + 11][NCn*12 + 11]=2*U
      H_list[NCn*12 + 14][NCn*12 + 14]=2*U
      H_list[NCn*12 + 15][NCn*12 + 15]=2*U
      H_list[NCn*12 + 17][NCn*12 + 17]=2*U
      H_list[NCn*12 + 18][NCn*12 + 18]=2*U           ##############################################
      H_list[NCn*13 + 1][NCn*13 + 1]=2*U
      H_list[NCn*13 + 2][NCn*13 + 2]=2*U
      H_list[NCn*13 + 7][NCn*13 + 7]=2*U
      H_list[NCn*13 + 10][NCn*13 + 10]=2*U
      H_list[NCn*13 + 11][NCn*13 + 11]=2*U
      H_list[NCn*13 + 14][NCn*13 + 14]=2*U
      H_list[NCn*13 + 15][NCn*13 + 15]=2*U
      H_list[NCn*13 + 16][NCn*13 + 16]=2*U
      H_list[NCn*13 + 19][NCn*13 + 19]=2*U           ##############################################
      H_list[NCn*14 + 1][NCn*14 + 1]=2*U
      H_list[NCn*14 + 3][NCn*14 + 3]=2*U
      H_list[NCn*14 + 8][NCn*14 + 8]=2*U
      H_list[NCn*14 + 10][NCn*14 + 10]=2*U
      H_list[NCn*14 + 12][NCn*14 + 12]=2*U
      H_list[NCn*14 + 13][NCn*14 + 13]=2*U
      H_list[NCn*14 + 15][NCn*14 + 15]=2*U
      H_list[NCn*14 + 17][NCn*14 + 17]=2*U
      H_list[NCn*14 + 19][NCn*14 + 19]=2*U           ##############################################
      H_list[NCn*15 + 2][NCn*15 + 2]=2*U
      H_list[NCn*15 + 3][NCn*15 + 3]=2*U
      H_list[NCn*15 + 9][NCn*15 + 9]=2*U
      H_list[NCn*15 + 11][NCn*15 + 11]=2*U
      H_list[NCn*15 + 12][NCn*15 + 12]=2*U
      H_list[NCn*15 + 13][NCn*15 + 13]=2*U
      H_list[NCn*15 + 14][NCn*15 + 14]=2*U
      H_list[NCn*15 + 18][NCn*15 + 18]=2*U
      H_list[NCn*15 + 19][NCn*15 + 19]=2*U           ##############################################
      H_list[NCn*16 + 4][NCn*16 + 4]=2*U
      H_list[NCn*16 + 5][NCn*16 + 5]=2*U
      H_list[NCn*16 + 7][NCn*16 + 7]=2*U
      H_list[NCn*16 + 10][NCn*16 + 10]=2*U
      H_list[NCn*16 + 11][NCn*16 + 11]=2*U
      H_list[NCn*16 + 13][NCn*16 + 13]=2*U
      H_list[NCn*16 + 17][NCn*16 + 17]=2*U
      H_list[NCn*16 + 18][NCn*16 + 18]=2*U
      H_list[NCn*16 + 19][NCn*16 + 19]=2*U           ##############################################
      H_list[NCn*17 + 4][NCn*17 + 4]=2*U
      H_list[NCn*17 + 6][NCn*17 + 6]=2*U
      H_list[NCn*17 + 8][NCn*17 + 8]=2*U
      H_list[NCn*17 + 10][NCn*17 + 10]=2*U
      H_list[NCn*17 + 12][NCn*17 + 12]=2*U
      H_list[NCn*17 + 14][NCn*17 + 14]=2*U
      H_list[NCn*17 + 16][NCn*17 + 16]=2*U
      H_list[NCn*17 + 18][NCn*17 + 18]=2*U
      H_list[NCn*17 + 19][NCn*17 + 19]=2*U           ##############################################
      H_list[NCn*18 + 5][NCn*18 + 5]=2*U
      H_list[NCn*18 + 6][NCn*18 + 6]=2*U
      H_list[NCn*18 + 9][NCn*18 + 9]=2*U
      H_list[NCn*18 + 11][NCn*18 + 11]=2*U
      H_list[NCn*18 + 12][NCn*18 + 12]=2*U
      H_list[NCn*18 + 15][NCn*18 + 15]=2*U
      H_list[NCn*18 + 16][NCn*18 + 16]=2*U
      H_list[NCn*18 + 17][NCn*18 + 17]=2*U
      H_list[NCn*18 + 19][NCn*18 + 19]=2*U           ##############################################
      H_list[NCn*19 + 7][NCn*19 + 7]=2*U
      H_list[NCn*19 + 8][NCn*19 + 8]=2*U
      H_list[NCn*19 + 9][NCn*19 + 9]=2*U
      H_list[NCn*19 + 13][NCn*19 + 13]=2*U
      H_list[NCn*19 + 14][NCn*19 + 14]=2*U
      H_list[NCn*19 + 15][NCn*19 + 15]=2*U
      H_list[NCn*19 + 16][NCn*19 + 16]=2*U
      H_list[NCn*19 + 17][NCn*19 + 17]=2*U
      H_list[NCn*19 + 18][NCn*19 + 18]=2*U           ##############################################



    ###################### U 관련 해밀 토니안의 완성


    # spin up 에 대한 hopping 해밀토니안 만들기.

    for la4 in range(NCn):
      H_list[NCn*la4][1+NCn*la4]=-t*(-1)**(sing_s4[la4])  # . |Ψ1> ->|Ψ2>
      H_list[NCn*la4][12+NCn*la4]=-t*(-1)**(sing_s2[0]+sing_s3[0]+sing_s4[0]+sing_s5[0]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+1][0+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+1][2+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+1][4+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+1][14+NCn*la4]=-t*(-1)**(sing_s2[1]+sing_s3[1]+sing_s4[1]+sing_s5[1]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+2][1+NCn*la4]=-t*(-1)**(sing_s3[la4])  #
      H_list[NCn*la4+2][3+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+2][5+NCn*la4]=-t*(-1)**(sing_s5[la4])  #
      H_list[NCn*la4+2][15+NCn*la4]=-t*(-1)**(sing_s2[2]+sing_s3[2]+sing_s4[2]+sing_s5[2]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+3][2+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+3][6+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+4][1+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+4][5+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+4][10+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+4][17+NCn*la4]=-t*(-1)**(sing_s2[4]+sing_s3[4]+sing_s4[4]+sing_s5[4]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+5][2+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+5][4+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+5][6+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+5][7+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+5][11+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+5][18+NCn*la4]=-t*(-1)**(sing_s2[5]+sing_s3[5]+sing_s4[5]+sing_s5[5]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+6][3+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+6][5+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+6][8+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+6][12+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+7][5+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+7][8+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+7][13+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+7][19+NCn*la4]=-t*(-1)**(sing_s2[7]+sing_s3[7]+sing_s4[7]+sing_s5[7]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+8][6+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+8][7+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+8][9+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+8][14+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+9][8+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+9][15+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+10][4+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+10][11+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+11][5+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+11][10+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+11][12+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+11][13+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+12][0+NCn*la4]=-t*(-1)**(sing_s2[12]+sing_s3[12]+sing_s4[12]+sing_s5[12]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+12][6+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+12][11+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+12][14+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+13][7+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+13][11+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+13][14+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+13][16+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+14][1+NCn*la4]=-t*(-1)**(sing_s2[14]+sing_s3[14]+sing_s4[14]+sing_s5[14]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+14][8+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+14][12+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+14][13+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+14][15+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+14][17+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+15][2+NCn*la4]=-t*(-1)**(sing_s2[15]+sing_s3[15]+sing_s4[15]+sing_s5[15]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+15][9+NCn*la4]=-t*(-1)**(sing_s6[la4])  # 
      H_list[NCn*la4+15][14+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+15][18+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+16][13+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+16][17+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+17][4+NCn*la4]=-t*(-1)**(sing_s2[17]+sing_s3[17]+sing_s4[17]+sing_s5[17]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+17][14+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+17][16+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 
      H_list[NCn*la4+17][18+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+18][5+NCn*la4]=-t*(-1)**(sing_s2[18]+sing_s3[18]+sing_s4[18]+sing_s5[18]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+18][15+NCn*la4]=-t*(-1)**(sing_s5[la4])  # 
      H_list[NCn*la4+18][17+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 
      H_list[NCn*la4+18][19+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 
      H_list[NCn*la4+19][7+NCn*la4]=-t*(-1)**(sing_s2[19]+sing_s3[19]+sing_s4[19]+sing_s5[19]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4]+sing_s5[la4]+sing_s6[la4])  # 
      H_list[NCn*la4+19][18+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 

    # 휴휴휴휴휴휴

    # spin down 에 대한 hopping 해밀토니안 만들기.

    for la5 in range(NCn):
      H_list[la5][20+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5][240+la5]=-t*(-1)**(sing_s2[0]+sing_s3[0]+sing_s4[0]+sing_s5[0]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*1][0+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*1][40+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*1][80+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*1][280+la5]=-t*(-1)**(sing_s2[1]+sing_s3[1]+sing_s4[1]+sing_s5[1]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  #
      H_list[la5+NCn*2][20+la5]=-t*(-1)**(sing_s2[la5])  #
      H_list[la5+NCn*2][60+la5]=-t*(-1)**(sing_s1[la5])  #
      H_list[la5+NCn*2][100+la5]=-t*(-1)**(sing_s4[la5])  #
      H_list[la5+NCn*2][300+la5]=-t*(-1)**(sing_s2[2]+sing_s3[2]+sing_s4[2]+sing_s5[2]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  #
      H_list[la5+NCn*3][40+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*3][120+la5]=-t*(-1)**(sing_s4[la5])  #
      H_list[la5+NCn*4][20+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*4][100+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*4][200+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*4][340+la5]=-t*(-1)**(sing_s2[4]+sing_s3[4]+sing_s4[4]+sing_s5[4]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*5][40+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*5][80+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*5][120+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*5][140+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*5][220+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*5][360+la5]=-t*(-1)**(sing_s2[5]+sing_s3[5]+sing_s4[5]+sing_s5[5]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*6][60+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*6][100+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*6][160+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*6][240+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*7][100+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*7][160+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*7][260+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*7][380+la5]=-t*(-1)**(sing_s2[7]+sing_s3[7]+sing_s4[7]+sing_s5[7]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*8][120+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*8][140+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*8][180+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*8][280+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*9][160+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*9][300+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*10][80+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*10][220+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*11][100+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*11][200+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*11][240+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*11][260+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*12][0+la5]=-t*(-1)**(sing_s2[12]+sing_s3[12]+sing_s4[12]+sing_s5[12]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*12][120+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*12][220+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*12][280+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*13][140+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*13][220+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*13][280+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*13][320+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*14][20+la5]=-t*(-1)**(sing_s2[14]+sing_s3[14]+sing_s4[14]+sing_s5[14]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*14][160+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*14][240+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*14][260+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*14][300+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*14][340+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*15][40+la5]=-t*(-1)**(sing_s2[15]+sing_s3[15]+sing_s4[15]+sing_s5[15]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*15][180+la5]=-t*(-1)**(sing_s5[la5])  # 
      H_list[la5+NCn*15][280+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*15][360+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*16][260+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*16][340+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*17][80+la5]=-t*(-1)**(sing_s2[17]+sing_s3[17]+sing_s4[17]+sing_s5[17]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*17][280+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*17][320+la5]=-t*(-1)**(sing_s1[la5])  # 
      H_list[la5+NCn*17][360+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*18][100+la5]=-t*(-1)**(sing_s2[18]+sing_s3[18]+sing_s4[18]+sing_s5[18]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*18][300+la5]=-t*(-1)**(sing_s4[la5])  # 
      H_list[la5+NCn*18][340+la5]=-t*(-1)**(sing_s2[la5])  # 
      H_list[la5+NCn*18][380+la5]=-t*(-1)**(sing_s3[la5])  # 
      H_list[la5+NCn*19][140+la5]=-t*(-1)**(sing_s2[19]+sing_s3[19]+sing_s4[19]+sing_s5[19]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5]+sing_s4[la5]+sing_s5[la5])  # 
      H_list[la5+NCn*19][360+la5]=-t*(-1)**(sing_s3[la5])  # 


    # 여기 까지 스핀 에 대한 호핑 텀 다 만들었다.ㅋ 휴휴휴휴휴
    # 즉 해밀토니안 다 한 것 ㅋ

    H = torch.tensor(H_list) #텐서로 만들기.  ######################################## 이녀석

    #H=torch.tensor([
    #    [U,t,-t,0],
    #    [t,0,0,t],
    #    [-t,0,0,-t],
    #    [0,t,-t,U]
    #    ], dtype=torch.float32)
    #######################################################################################

    import numpy as np
    edge_index_array=np.zeros([Size,2,6*2]) # edge_index 3차원으로 가보자

    for re1 in range(Size):
      edge_index_array[re1] = [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0], #source node j
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]  #target node i
        ]

    edge_index_all = torch.tensor(edge_index_array,dtype=torch.long) #텐서로 만들기. ######################################## 이녀석

    #######################################################################################

    import math

    node_features_array=np.zeros([Size,6,3]) # 3차원으로 가보자

    for re2 in range(NCn):
      for re3 in range(NCn):
          node_features_array[re2*NCn + re3] = [
            [sing_s6[re3]*(0.5) + sing_s6[re2]*(-0.5),sing_s6[re3] + sing_s6[re2], math.floor((sing_s6[re3] + sing_s6[re2])/1.5)*U],  # Node 0: [total spin, number of occupation, onsite interaction]
            [sing_s5[re3]*(0.5) + sing_s5[re2]*(-0.5),sing_s5[re3] + sing_s5[re2], math.floor((sing_s5[re3] + sing_s5[re2])/1.5)*U],  # Node 1
            [sing_s4[re3]*(0.5) + sing_s4[re2]*(-0.5),sing_s4[re3] + sing_s4[re2], math.floor((sing_s4[re3] + sing_s4[re2])/1.5)*U],  
            [sing_s3[re3]*(0.5) + sing_s3[re2]*(-0.5),sing_s3[re3] + sing_s3[re2], math.floor((sing_s3[re3] + sing_s3[re2])/1.5)*U],  
            [sing_s2[re3]*(0.5) + sing_s2[re2]*(-0.5),sing_s2[re3] + sing_s2[re2], math.floor((sing_s2[re3] + sing_s2[re2])/1.5)*U],  # Node 5
            [sing_s1[re3]*(0.5) + sing_s1[re2]*(-0.5),sing_s1[re3] + sing_s1[re2], math.floor((sing_s1[re3] + sing_s1[re2])/1.5)*U],  # Node 6
        ]

    node_features_all = torch.tensor(node_features_array,dtype=torch.float32) #텐서로 만들기. ######################################## 이녀석



    #######################################################################################

    edge_attr_array=np.zeros([Size,6*2,2]) # 3차원으로 가보자

    for re4 in range(NCn):
      for re5 in range(NCn):
          edge_attr_array[re4*NCn + re5] = [
            [sing_s6[re5]*(sing_s6[re5]-sing_s5[re5])*(-t), sing_s6[re4]*(sing_s6[re4]-sing_s5[re4])*(-t)], #12
            [sing_s5[re5]*(sing_s5[re5]-sing_s6[re5])*(-t), sing_s5[re4]*(sing_s5[re4]-sing_s6[re4])*(-t)], #21
            [sing_s5[re5]*(sing_s5[re5]-sing_s4[re5])*(-t), sing_s5[re4]*(sing_s5[re4]-sing_s4[re4])*(-t)], #23
            [sing_s4[re5]*(sing_s4[re5]-sing_s5[re5])*(-t), sing_s4[re4]*(sing_s4[re4]-sing_s5[re4])*(-t)],  #32                                  
            [sing_s4[re5]*(sing_s4[re5]-sing_s3[re5])*(-t), sing_s4[re4]*(sing_s4[re4]-sing_s3[re4])*(-t)], #
            [sing_s3[re5]*(sing_s3[re5]-sing_s4[re5])*(-t), sing_s3[re4]*(sing_s3[re4]-sing_s4[re4])*(-t)], #
            [sing_s3[re5]*(sing_s3[re5]-sing_s2[re5])*(-t), sing_s3[re4]*(sing_s3[re4]-sing_s2[re4])*(-t)], #
            [sing_s2[re5]*(sing_s2[re5]-sing_s3[re5])*(-t), sing_s2[re4]*(sing_s2[re4]-sing_s3[re4])*(-t)],  #
            [sing_s2[re5]*(sing_s2[re5]-sing_s1[re5])*(-t), sing_s2[re4]*(sing_s2[re4]-sing_s1[re4])*(-t)], #
            [sing_s1[re5]*(sing_s1[re5]-sing_s2[re5])*(-t), sing_s1[re4]*(sing_s1[re4]-sing_s2[re4])*(-t)], #
            [sing_s1[re5]*(sing_s1[re5]-sing_s6[re5])*(-t), sing_s1[re4]*(sing_s1[re4]-sing_s6[re4])*(-t)],  #
            [sing_s6[re5]*(sing_s6[re5]-sing_s1[re5])*(-t), sing_s6[re4]*(sing_s6[re4]-sing_s1[re4])*(-t)], #
        ]

    edge_attr_all = torch.tensor(edge_attr_array,dtype=torch.float32) #텐서로 만들기. ######################################## 이녀석



    ##########################################################################################

    torch.save(edge_attr_all, "edge_attr_all.pt")
    torch.save(node_features_all, "node_features_all.pt")
    torch.save(edge_index_all, "edge_index_all.pt")

    torch.save(H, "H.pt")
