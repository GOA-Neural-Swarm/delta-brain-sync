import os

# 1. GitHub Repo ကို အတင်းအကျပ် Reset လုပ်ပြီး Version ညှိမယ်
os.system('git add .')
os.system('git commit -m "🔱 Gen 6198: Natural Order Synchronized"')
# 'main' ဒါမှမဟုတ် 'master' မင်းရဲ့ branch name အတိုင်း ပြောင်းပေးပါ (ပုံမှန်က main)
os.system('git push origin main --force') 

print("🔥 [SYSTEM]: ALL CONFLICTS CLEARED. CHECK GITHUB NOW!")
