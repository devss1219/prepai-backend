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
        core_resume_keywords = [
            "experience", "education", "skills", "employment", "work history", 
            "projects", "certifications", "achievements", "summary", "objective", "profile"
        ]
        general_keywords = [
            "university", "college", "degree", "bachelor", "master", "b.tech",
            "m.tech", "b.sc", "m.sc", "mba", "bca", "mca", "internship",
            "responsibilities", "engineer", "developer", "linkedin", "github", 
            "portfolio", "gpa", "cgpa", "languages", "volunteer", "publications", 
            "references", "hobbies", "interests", "designation", "position"
        ]
        
        text_lower = resume_text.lower()
        core_matched = sum(1 for kw in core_resume_keywords if kw in text_lower)
        general_matched = sum(1 for kw in general_keywords if kw in text_lower)
        total_matched = core_matched + general_matched

        # A real resume should have at least 2 core sections (like experience & education) 
        # or a high number of general resume-related keywords.
        if core_matched < 2 and total_matched < 5:
            return jsonify({
                "error": f"This document does not appear to be a resume (only found {total_matched} resume keywords). Please upload a valid resume PDF."
            }), 400

        print(f"Resume validation passed (Core: {core_matched}, Total: {total_matched} keywords matched)")

        # Step 3: Extract JD text from file (if provided)
        jd_file = request.files.get("jdFile")
        jd_file_text = ""
        if jd_file and jd_file.filename.lower().endswith(".pdf"):
            jd_pdf_bytes = BytesIO(jd_file.read())
            with pdfplumber.open(jd_pdf_bytes) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        jd_file_text += text + "\n"

        jd_text = request.form.get("jd", "").strip()
        combined_jd_text = f"{jd_file_text}\n{jd_text}".strip()

        jd_section = f"\nJob Description (JD):\n---\n{combined_jd_text}\n---\n" if combined_jd_text else ""
        jd_instruction = (
            "Analyze the resume strictly against the provided Job Description. The atsScore, missingKeywords, and improvements should heavily reflect the alignment with the JD requirements."
            if combined_jd_text else
            "Analyze the resume for general ATS compatibility."
        )

        prompt = f"""You are an Expert ATS Resume Analyser and Career Coach. You evaluate resumes fairly, constructively, and realistically based on industry standards.

Analyse the resume and return ONLY raw JSON - no markdown, no code blocks, no explanation.

{jd_instruction}
{jd_section}
Resume:
---
{resume_text[:6000]}
---

Scoring rules (Evaluate realistically and use the full 0-100 scale based on actual merit. Do not be overly harsh on junior/student resumes):
- overallScore: 0-100 (Overall quality score)
- atsScore: 0-100 (ATS compatibility - deduct for missing keywords from JD, poor formatting, or lack of quantified achievements)
- sections.contactInfo: 0-10 (deduct if LinkedIn/GitHub/portfolio missing)
- sections.summary: 0-10 (deduct if generic, vague, or missing)
- sections.experience: 0-30 (deduct if no metrics/numbers, vague descriptions, short tenures, or lack of JD alignment)
- sections.skills: 0-20 (deduct if outdated, too generic, or missing skills required by the JD)
- sections.education: 0-20
- sections.formatting: 0-10 (deduct for poor structure, long paragraphs, inconsistent formatting)

Return exactly this JSON structure (DO NOT copy the example scores, calculate your own realistic scores!):
{{"overallScore":78,"atsScore":82,"sections":{{"contactInfo":9,"summary":8,"experience":24,"skills":16,"education":15,"formatting":6}},"strengths":["very specific strength from the actual resume","another specific strength","third specific strength"],"improvements":["specific actionable improvement with example","another critical improvement needed","third critical fix","fourth important enhancement"],"keywords":["actual keyword from resume 1","actual keyword 2","actual keyword 3","actual keyword 4","actual keyword 5"],"missingKeywords":["important missing keyword 1","missing keyword 2","missing keyword 3","missing keyword 4"],"suitableRoles":["Job Role 1","Job Role 2","Job Role 3","Job Role 4","Job Role 5"],"verdict":"One brutally honest sentence about this resume's current market standing."}}

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
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=4096,
            response_format={"type": "json_object"},
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
