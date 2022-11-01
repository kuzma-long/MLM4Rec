# 扩充训练集
def aug_p(i_file,o_file,thresh):
    """
    :param i_file:待处理的数据集
    :param o_file: 处理后保存的数据集
    :return:
    """
    with open(i_file,"r+") as fr:
        data=fr.readlines()
    aug_d={}
    for d_ in data:
        u_i,item=d_.split(' ',1)
        item=item.split(' ')
        item[-1]=str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start=0
        if len(item)>thresh:
            while start<len(item)-thresh-1:
                aug_d[u_i].append(item[start:start+thresh])
                start+=1
        elif len(item)>4:
            aug_d[u_i].append(item)
    with open(o_file,"w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i+" "+' '.join(i_)+"\n")
# def aug_p(i_file,o_file):
#     """
#     :param i_file:待处理的数据集
#     :param o_file: 处理后保存的数据集
#     :return:
#     """
#     with open(i_file,"r+") as fr:
#         data=fr.readlines()
#     aug_d={}
#     for d_ in data:
#         u_i,item=d_.split(' ',1)
#         item=item.split(' ')
#         item[-1]=str(eval(item[-1]))
#         aug_d.setdefault(u_i, [])
#         start=0
#         j=4
#         if len(item)>50:
#             while start<len(item)-49:
#                 j=start+5
#                 while j<len(item):
#                     if start<1 and j-start<51:
#                         aug_d[u_i].append(item[start:j])
#                         j+=1
#                     else:
#                         aug_d[u_i].append(item[start:start+50])
#                         break
#                 start+=1
#         else:
#             while j<len(item):
#                 aug_d[u_i].append(item[start:j+1])
#                 j+=1
#     with open(o_file,"w+") as fw:
#         for u_i in aug_d:
#             for i_ in aug_d[u_i]:
#                 fw.write(u_i+" "+' '.join(i_)+"\n")