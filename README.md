## Dataset
קישור למערך הנתונים  [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new) available on Kaggle.

לפני שמתחילים: בחירת סביבת עבוד
( ההסבר אינו לתכנת המתחיל אלא למפתחים שרוצים להמשיך עם הקוד)
הקוד נכתב בפייתון – דרושה גירסה 3.8 ומעלה
ולפני שמתחילים  חשוב לבחור את סביבת העבודה שבה תעבוד. במדריך זה נתמקד בשתי סביבות עיקריות
Visual Studio Code (VS Code):
מתאימה לעבודה מקומית על המחשב האישי, עם שליטה מלאה על הקבצים, הסביבה הזו נוחה להרצות טסטים מהירות ותיקונים מבלי להסתמך על קצב האינטרנט ומעבדים חיצוניים .
כדי להתחיל בויזואל צריך להתקין את הספריות הנ"ל
חבילה
למה צריך אותה
tensorflow
להרצת מודל -VGG16, לבניית והפעלת רשתות נוירונים
pandas
ניתוח נתונים וקריאת קבצים
scikit-learn
פיצול נתונים 
matplotlib
הצגת גרפים של דיוק/איבוד במהלך האימון
opencv-python
עיבוד תמונה, טשטוש פנים
Pillow
טיפול בתמונות בפורמט GUI
kagglehub
להורדת הנתונים (דטה)-Kaggle
retinaface
לאיתור פנים בתמונה (לצורך טשטוש מדויק)
tk (tkinter)
יצירת GUI לבחירת תמונות והצגת תוצאות
numpy
חישובים מתמטיים על מערכים


ההתקנה פשוטה – הרצה ב   

CMD  
pip install tensorflow
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install opencv-python
pip install pillow
pip install kagglehub
pip install retinaface

ניתן להריץ הכל בbat   

----------------------------------------------STARTBAT
חשוב להריץ במשתמש אדמיניסטרטור כדי לקבל הרשאות מתאימות:
@echo off
echo =====================

python -m 
venv .venv


call .venv\Scripts\activate


python -m pip install --upgrade pip


pip install tensorflow
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install opencv-python
pip install pillow
pip install kagglehub
pip install retinaface

echo ====================
echo Installation complete.
echo To activate the environment manually:
echo     call .venv\Scripts\activate
pause
   -----------------------------------------------------------------------------------------------------END BAT
לאחר אפשר לראות את התוצאות והלוגים 

View the Results on TensorBoard
cmd tensorboard --logdir="D:\projectResults\logs"
http://localhost:6006

Google Colab.
מתאימה לעבודה מהירה בענן, ללא צורך בהתקנות, עם גישה לולשיתוף קל עם אחרים.
 בשתי הסביבות במקביל, אבל חשוב לדעת שלכל אחת יש דרישות וכלים משלה. מעבר מסביבה לסביבה דורש התאמות קטנות – למשל, איך מתקינים ספריות, או איך טוענים קבצים
.
https://colab.research.google.com
התקנת ספריות בקולאב
!pip install kagglehub
!pip install retina-face
!pip install opencv-python
!pip install Pillow
!pip install scikit-learn
!pip install matplotlib
או
!pip install tensorflow keras pandas opencv-python matplotlib scikit-learn pillow
העלאת קבצים
python
from google.colab import files
uploaded = files.upload()
הורדת קבצים
python
from google.colab import files
files.download("output.jpg")

