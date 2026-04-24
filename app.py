import os
import json
import re
import pdfplumber
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

app = Flask(__name__)
CORS(app)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
api_key_status = "YES" if os.getenv("GROQ_API_KEY") else "NO"
print(f"GROQ_API_KEY loaded: {api_key_status}")


@app.route("/", methods=["GET"])
def health():
    return "PrepAI Resume Enhancer Backend (Python + Groq) Running!"


@app.route("/upload", methods=["POST"])
def upload():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        # Step 1: Extract text using pdfplumber
        pdf_bytes = BytesIO(file.read())
        resume_text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    resume_text += text + "\n"

        resume_text = resume_text.strip()
        print(f"Extracted text length: {len(resume_text)}")

        if len(resume_text) < 50:
            return jsonify({
                "error": "Could not extract text from PDF. Make sure it is not a scanned image."
            }), 400

        # Step 2: Validate it is actually a resume
        resume_keywords = [
            "experience", "education", "skills", "work history", "employment",
            "university", "college", "degree", "bachelor", "master", "b.tech",
            "m.tech", "b.sc", "m.sc", "mba", "bca", "mca", "internship",
            "projects", "certifications", "objective", "summary", "profile",
            "achievements", "responsibilities", "engineer", "developer",
            "linkedin", "github", "portfolio", "gpa", "cgpa", "languages",
            "volunteer", "publications", "references", "hobbies", "interests",
            "phone", "email", "address", "contact", "designation", "position"
        ]
        text_lower = resume_text.lower()
        matched = sum(1 for kw in resume_keywords if kw in text_lower)

        if matched < 2:
            return jsonify({
                "error": "This does not appear to be a resume. Please upload a valid resume PDF."
            }), 400

        print(f"Resume validation passed ({matched} keywords matched)")

        prompt = f"""You are a STRICT ATS Resume Analyser and Career Coach with 20 years of experience. You do NOT give inflated scores. Be brutally honest and critical.

Analyse the resume and return ONLY raw JSON - no markdown, no code blocks, no explanation.

Resume:
---
{resume_text[:6000]}
---

Scoring rules (be STRICT - average resumes should score 40-60, only exceptional ones above 80):
- overallScore: 0-100 (harsh overall quality score)
- atsScore: 0-100 (ATS compatibility - penalise heavily for missing keywords, poor formatting, no quantified achievements)
- sections.contactInfo: 0-10 (deduct if LinkedIn/GitHub/portfolio missing)
- sections.summary: 0-10 (deduct if generic, vague, or missing)
- sections.experience: 0-30 (deduct heavily if no metrics/numbers, vague descriptions, short tenures)
- sections.skills: 0-20 (deduct if outdated, too generic, or not matched to experience)
- sections.education: 0-20
- sections.formatting: 0-10 (deduct for poor structure, long paragraphs, inconsistent formatting)

Return exactly this JSON:
{{"overallScore":55,"atsScore":48,"sections":{{"contactInfo":7,"summary":5,"experience":18,"skills":12,"education":16,"formatting":7}},"strengths":["very specific strength from the actual resume","another specific strength","third specific strength"],"improvements":["specific actionable improvement with example","another critical improvement needed","third critical fix","fourth important enhancement"],"keywords":["actual keyword from resume 1","actual keyword 2","actual keyword 3","actual keyword 4","actual keyword 5"],"missingKeywords":["important missing keyword 1","missing keyword 2","missing keyword 3","missing keyword 4"],"suitableRoles":["Job Role 1","Job Role 2","Job Role 3","Job Role 4","Job Role 5"],"verdict":"One brutally honest sentence about this resume's current market standing."}}

For suitableRoles: list 5 specific job titles this resume is genuinely qualified for RIGHT NOW based on actual experience and skills shown. Be realistic, not aspirational.

Return ONLY the JSON object."""

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ATS Resume Analyser. You only respond with raw JSON, no markdown, no extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )

        raw_text = chat_completion.choices[0].message.content.strip()
        print(f"Groq raw (first 300): {raw_text[:300]}")

        # Strip markdown fences if present
        raw_text = re.sub(r'^```[\w]*\n?', '', raw_text)
        raw_text = re.sub(r'\n?```$', '', raw_text).strip()

        analysis = json.loads(raw_text)

        return jsonify({
            "success": True,
            "fileName": file.filename,
            "analysis": analysis
        })

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw text was: {raw_text if 'raw_text' in dir() else 'N/A'}")
        return jsonify({"error": "AI returned invalid response. Try again.", "detail": str(e)}), 500
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({"error": "AI analysis failed. Try again.", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
