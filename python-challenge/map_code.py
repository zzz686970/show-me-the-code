import string
msg = "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj."

original = list("".join('abcdefghijklmnopqrstuvwxyz'))

encrypt = list("".join('cdefghijklmnopqrstuvwxyzab'))

table = string.maketrans('abcdefghijklmnopqrstuvwxyz', 'cdefghijklmnopqrstuvwxyzab')

print(msg.translate(table))

# dict_note ={}
# for k, v in zip(original, encrypt):
# 	dict_note[k] = v
# decode_msg = []
# new_str = ""
# for ele in msg:
# 	if ele in dict_note.keys():
# 		decode_msg.append(dict_note[ele])
# 	else:
# 		decode_msg.append(ele)

# for ele in decode_msg:
# 	new_str += ele
# print(new_str)


