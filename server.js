import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { createRequire } from 'module';
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

// Manually load .env (avoids dotenv import order issues with ESM)
const __dirname = dirname(fileURLToPath(import.meta.url));
try {
  const envFile = readFileSync(resolve(__dirname, '.env'), 'utf8');
  envFile.split('\n').forEach(line => {
    const [key, ...rest] = line.split('=');
    if (key && rest.length) process.env[key.trim()] = rest.join('=').trim();
  });
} catch (e) { /* no .env file — use system env */ }

const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');

console.log('GEMINI_API_KEY loaded:', process.env.GEMINI_API_KEY ? 'YES ✅' : 'NO ❌');

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') cb(null, true);
    else cb(new Error('Only PDF files are allowed'), false);
  }
});

// --- RESUME UPLOAD & ANALYSE ROUTE ---
app.post('/upload', upload.single('resume'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  try {
    // Step 1: Extract text from PDF
    const pdfData = await pdfParse(req.file.buffer);
    const resumeText = pdfData.text.trim();
    console.log('Extracted text length:', resumeText.length);

    if (!resumeText || resumeText.length < 50) {
      return res.status(400).json({ error: 'Could not extract text from PDF. Make sure it is not a scanned image.' });
    }

    // Step 2: Gemini Analysis
    const prompt = `You are an expert ATS Resume Analyser and Career Coach. Analyse the following resume and return ONLY raw JSON — no markdown, no code blocks, no explanation.

Resume:
"""
${resumeText.slice(0, 6000)}
"""

Return exactly this JSON (fill in real values based on the resume):
{"overallScore":75,"atsScore":68,"sections":{"contactInfo":8,"summary":7,"experience":22,"skills":15,"education":16,"formatting":8},"strengths":["strength1","strength2","strength3"],"improvements":["improvement1","improvement2","improvement3","improvement4"],"keywords":["kw1","kw2","kw3","kw4","kw5"],"missingKeywords":["mk1","mk2","mk3","mk4"],"verdict":"One sentence verdict."}`;

    const response = await ai.models.generateContent({
      model: 'gemini-2.0-flash',
      contents: prompt,
    });

    let rawText = response.text.trim();
    console.log('Gemini raw (first 300):', rawText.slice(0, 300));

    // Strip markdown fences if present
    rawText = rawText.replace(/^```[\w]*\n?/, '').replace(/\n?```$/, '').trim();

    const analysis = JSON.parse(rawText);

    res.json({ success: true, fileName: req.file.originalname, analysis });

  } catch (err) {
    console.error('Analysis error:', err.message);
    res.status(500).json({ error: 'AI analysis failed. Try again.', detail: err.message });
  }
});

app.get('/', (req, res) => res.send('PrepAI Resume Enhancer Backend Running 🚀'));

app.listen(PORT, () => console.log(`Server on http://localhost:${PORT}`));