import streamlit as st
import re
import pickle
import nltk


nltk.download('punkt')  # this work in backend
nltk.download('stopwords')  # this work in backend

# loading models
clf=pickle.load(open('clf.pkl','rb'))
tfidfd=pickle.load(open('tfidf.pkl','rb'))


def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Web app
def main():
    st.title("Resume Screening App")
    uploaded_file=st.file_uploader('Upload Resume',type=['pdf','txt'])   # the file can be txt and pdf.


    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()    # read file in bytes
            resume_text=resume_bytes.decode('utf-8')    #check the decode bytes in "utf-8" and return in resume_text
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text=resume_bytes.decode('latin-1')

        cleaned_resume=clean_resume(resume_text)    # clean the resume and store in variable cleaned_resume
        input_features=tfidfd.transform([cleaned_resume])   # cleaned_resume transformed in vectorization and store in input_feature
        prediction_id=clf.predict(input_features)[0]        # based on input_feature predict the prediction id
        st.write(prediction_id)          # print prediction_id

         # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)


# Python main
if __name__=="__main__":
    main()    