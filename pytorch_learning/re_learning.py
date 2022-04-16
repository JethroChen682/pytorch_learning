import re
text=""
file=open("poem.txt")

for line in file:
    text=text+line
result=re.findall(" (a[a-z][a-z])[ .]|(A[a-z][a-z])[ .]",text)
final_result=set()
for pair in result:
    final_result.add(pair[0])
    final_result.add(pair[1])
final_result.remove("")
file.close()
result=re.findall("\w{1,3}",text)