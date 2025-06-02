תקנה מהירה (מומלץ)
ניתן להשתמש בקובץ אצווה אחד בלבד שמבצע את כל תהליך ההתקנה:

install_full_env.bat

מה הוא עושה?

מתקין את כל הספריות הדרושות

יוצר סביבה וירטואלית (venv)

מקשר את הסביבה ל־VS Code

פותח את Visual Studio Code עם הסביבה מוכנה

הורד את הקובץ, לחץ עליו לחיצה ימנית ובחר "Run as administrator"

התקנה ידנית – שלב אחר שלב
למי שמעדיף שליטה מלאה, ניתן להשתמש בשלושת הקבצים הבאים לפי הסדר:

install_and_open_vsc.bat
פותח VS Code ומכין את הסביבה.

setup_env.bat

יוצרת סביבה וירטואלית

מפעילה אותה

מתקינה את כל הספריות לפי הקובץ הבא:

requirements.txt
כולל את רשימת כל הספריות + הגדרות גרסאות מדויקות.

| ספרייה           | תיאור שימוש                         |
| ---------------- | ----------------------------------- |
| `tensorflow`     | הרצת מודל VGG16 לבניית רשת נוירונים |
| `pandas`         | עיבוד נתונים, קריאת CSV             |
| `scikit-learn`   | פיצול מערכי נתונים                  |
| `matplotlib`     | גרפים וויזואליזציה של תוצאות        |
| `opencv-python`  | עיבוד תמונה, כולל טשטוש פנים        |
| `pillow`         | טיפול בתמונות בפורמט GUI            |
| `kagglehub`      | הורדת מודלים ודאטה מ־Kaggle         |
| `retinaface`     | זיהוי פנים לצורך טשטוש              |
| `tk` / `tkinter` | ממשק גרפי לבחירת תמונה              |
| `numpy`          | חישובים מתמטיים על מערכים           |


@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip

pip install tensorflow==2.14.0
pip install keras==2.14.0
pip install pandas==2.2.3
pip install scikit-learn==1.6.1
pip install matplotlib==3.10.3
pip install opencv-python==4.11.0.86
pip install pillow==11.2.1
pip install kagglehub==0.3.12
pip install retina-face==0.0.17
pip install flask==3.1.0
pip install tqdm==4.67.1
pip install protobuf==4.25.7
pip install tensorboard==2.14.1
pip install gdown==5.2.0

echo ====================
echo All packages installed successfully.
pause
