import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# โหลดข้อมูล
df = pd.read_csv("data/earthquakes.csv")

# แปลงค่าตัวหนังสือเป็นตัวเลข (alert)
le = LabelEncoder()
df["alert"] = le.fit_transform(df["alert"])

# เลือก features และ target
X = df[["magnitude", "depth", "cdi", "mmi", "sig"]]
y = df["alert"]

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและเทรนโมเดล
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ประเมินผล
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# บันทึกโมเดล
joblib.dump(model, "earthquake_model.pkl")
print("\n💾 บันทึกโมเดลเรียบร้อย: earthquake_model.pkl")

# บันทึก encoder
joblib.dump(le, "label_encoder.pkl")
print("💾 บันทึกตัวแปลง LabelEncoder เรียบร้อย: label_encoder.pkl")
