from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from flask_cors import CORS
import io
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import sqlite3
from datetime import datetime, timedelta
import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:8081"]}}
)
# Configuration
class Config:
    DATABASE_PATH = 'employee_faces.db'
    FACE_MODEL = "resnet50_2020-07-20"
    MAX_IMAGE_SIZE = 1048
    DEVICE = "cpu"  # Change to "cuda" if GPU available
    SIMILARITY_THRESHOLD = 0.5
    MAX_FACES_PER_IMAGE = 10

config = Config()

class FaceRecognitionSystem:
    def __init__(self):
        self.model = None
        self.load_model()
        self.init_database()
    
    def load_model(self):
        """Load the RetinaFace model"""
        try:
            self.model = get_model(
                config.FACE_MODEL, 
                max_size=config.MAX_IMAGE_SIZE, 
                device=config.DEVICE
            )
            self.model.eval()
            logger.info("RetinaFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    # def init_database(self):
    #     """Initialize SQLite database for employee faces and leave requests"""
    #     conn = sqlite3.connect(config.DATABASE_PATH)
    #     cursor = conn.cursor()
        
    #     # Employees table
    #     cursor.execute('''
    #         CREATE TABLE IF NOT EXISTS employees (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             employee_id TEXT UNIQUE NOT NULL,
    #             name TEXT NOT NULL,
    #             department TEXT,
    #             position TEXT,
    #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    #         )
    #     ''')
        
    #     # Face embeddings table
    #     cursor.execute('''
    #         CREATE TABLE IF NOT EXISTS face_embeddings (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             employee_id TEXT NOT NULL,
    #             embedding BLOB NOT NULL,
    #             image_path TEXT,
    #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #             FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    #         )
    #     ''')
        
    #     # Attendance logs table
    #     cursor.execute('''
    #         CREATE TABLE IF NOT EXISTS attendance_logs (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             employee_id TEXT NOT NULL,
    #             check_in_time TIMESTAMP,
    #             check_out_time TIMESTAMP,
    #             date DATE,
    #             status TEXT DEFAULT 'present',
    #             confidence REAL,
    #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #             FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    #         )
    #     ''')
        
    #     # Leave requests table
    #     cursor.execute('''
    #         CREATE TABLE IF NOT EXISTS leave_requests (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             employee_id TEXT NOT NULL,
    #             leave_type TEXT NOT NULL,
    #             start_date DATE NOT NULL,
    #             end_date DATE NOT NULL,
    #             reason TEXT,
    #             status TEXT DEFAULT 'pending',
    #             submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #             approved_at TIMESTAMP,
    #             approved_by TEXT,
    #             rejection_reason TEXT,
    #             FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    #         )
    #     ''')
        
    #     conn.commit()
    #     conn.close()
    #     logger.info("Database initialized successfully")

    def init_database(self):
        """Initialize SQLite database for employee faces, attendance, and leave requests"""
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT,
                position TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Face embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        # Attendance logs table (with GPS support)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                check_in_time TIMESTAMP,
                check_out_time TIMESTAMP,
                date DATE,
                status TEXT DEFAULT 'present',
                confidence REAL,
                latitude REAL,
                longitude REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        # Leave requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leave_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                leave_type TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_at TIMESTAMP,
                approved_by TEXT,
                rejection_reason TEXT,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    
    
    def extract_face_features(self, image: np.ndarray) -> List[Dict]:
        """Extract face features using RetinaFace"""
        try:
            with torch.no_grad():
                annotations = self.model.predict_jsons(image)
            
            if not annotations or not annotations[0].get("bbox"):
                return []
            
            faces = []
            for annotation in annotations[:config.MAX_FACES_PER_IMAGE]:
                bbox = annotation["bbox"]
                landmarks = annotation.get("landmarks", [])
                
                # Extract face region
                x1, y1, x2, y2 = map(int, bbox)
                face_region = image[y1:y2, x1:x2]
                
                # Generate face embedding (simplified - in production use FaceNet or similar)
                face_embedding = self._generate_face_embedding(face_region)
                
                faces.append({
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "embedding": face_embedding,
                    "face_region": face_region
                })
            
            return faces
        except Exception as e:
            logger.error(f"Failed to extract face features: {str(e)}")
            return []
    
    def _generate_face_embedding(self, face_region: np.ndarray) -> np.ndarray:
        """Generate face embedding (simplified version)"""
        try:
            if face_region.size == 0:
                return np.zeros(512)
            
            # Resize face to standard size
            face_resized = cv2.resize(face_region, (112, 112))
            
            # Convert to grayscale and normalize
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            else:
                face_gray = face_resized
            
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Create a simple embedding using image patches
            embedding = []
            patch_size = 8
            for i in range(0, 112, patch_size):
                for j in range(0, 112, patch_size):
                    patch = face_normalized[i:i+patch_size, j:j+patch_size]
                    embedding.extend([
                        np.mean(patch),
                        np.std(patch),
                        np.min(patch),
                        np.max(patch)
                    ])
            
            return np.array(embedding[:512])  # Ensure fixed size
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return np.zeros(512)
    
    def register_employee(self, employee_data: dict, image) -> dict:
        """Register a new employee with their face"""
        try:
            # If image is bytes, convert to PIL.Image
            if isinstance(image, bytes):
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(image))

            # Extract face features directly from the raw image
            faces = self.extract_face_features(image)

            if not faces:
                return {"success": False, "message": "No face detected in image"}

            if len(faces) > 1:
                return {"success": False, "message": "Multiple faces detected. Please use image with single face"}

            face = faces[0]
            embedding = face["embedding"]

            # Store in database
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()

            # Insert employee details
            cursor.execute('''
                INSERT OR REPLACE INTO employees (employee_id, name, department, position)
                VALUES (?, ?, ?, ?)
            ''', (
                employee_data["employee_id"],
                employee_data["name"],
                employee_data.get("department", ""),
                employee_data.get("position", "")
            ))

            # Insert face embedding as blob
            embedding = face["embedding"].astype(np.float32) 
            embedding_blob = embedding.tobytes()
            cursor.execute('''
                INSERT INTO face_embeddings (employee_id, embedding)
                VALUES (?, ?)
            ''', (employee_data["employee_id"], embedding_blob))

            conn.commit()
            conn.close()

            return {
                "success": True,
                "message": f"Employee {employee_data['name']} registered successfully",
                "employee_id": employee_data["employee_id"]
            }

        except Exception as e:
            logger.error(f"Failed to register employee: {str(e)}")
            return {"success": False, "message": str(e)}
 
    def recognize_employee(self, image) -> dict:
        """Recognize employee(s) from a raw image"""
        try:
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert("RGB")

            if isinstance(image, Image.Image):
                image = np.array(image)

            # Extract face features
            faces = self.extract_face_features(image)

            if not faces:
                return {"success": False, "message": "No face detected"}

            # Get all registered embeddings
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT fe.employee_id, fe.embedding, e.name
                FROM face_embeddings fe
                JOIN employees e ON fe.employee_id = e.employee_id
            ''')
            registered_faces = cursor.fetchall()
            conn.close()

            if not registered_faces:
                return {"success": False, "message": "No registered employees found"}

            best_matches = []
            for face in faces:
                query_embedding = face["embedding"]
                best_similarity = 0
                best_match = None

                for emp_id, embedding_blob, name in registered_faces:
                    stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    similarity = self._calculate_similarity(query_embedding, stored_embedding)

                    if similarity > best_similarity and similarity > config.SIMILARITY_THRESHOLD:
                        best_similarity = similarity
                        best_match = {
                            "employee_id": emp_id,
                            "name": name,
                            "confidence": float(similarity),
                            "bbox": face["bbox"]
                        }

                if best_match:
                    best_matches.append(best_match)

            return {
                "success": True,
                "matches": best_matches,
                "total_faces": len(faces),
                "recognized_faces": len(best_matches)
            }

        except Exception as e:
            logger.error(f"Failed to recognize employee: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings (float32 safe)"""
        try:
            if embedding1.shape != embedding2.shape:
                logger.error(f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}")
                return 0.0

            emb1 = embedding1.astype(np.float32)
            emb2 = embedding2.astype(np.float32)

            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    # def log_attendance(self, employee_id: str, action: str, confidence: float = 0.0) -> Dict:
    #     """Log attendance (check-in/check-out)"""
    #     try:
    #         conn = sqlite3.connect(config.DATABASE_PATH)
    #         cursor = conn.cursor()
            
    #         current_time = datetime.now()
    #         current_date = current_time.date()
            
    #         if action == "check_in":
    #             cursor.execute('''
    #                 SELECT id FROM attendance_logs 
    #                 WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL
    #             ''', (employee_id, current_date))
                
    #             if cursor.fetchone():
    #                 conn.close()
    #                 return {"success": False, "message": "Already checked in today"}
                
    #             cursor.execute('''
    #                 INSERT INTO attendance_logs (employee_id, check_in_time, date, confidence)
    #                 VALUES (?, ?, ?, ?)
    #             ''', (employee_id, current_time, current_date, confidence))
                
    #         elif action == "check_out":
    #             cursor.execute('''
    #                 SELECT id FROM attendance_logs 
    #                 WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL AND check_out_time IS NULL
    #             ''', (employee_id, current_date))
                
    #             record = cursor.fetchone()
    #             if not record:
    #                 conn.close()
    #                 return {"success": False, "message": "No check-in record found for today"}
                
    #             cursor.execute('''
    #                 UPDATE attendance_logs 
    #                 SET check_out_time = ?, confidence = ?
    #                 WHERE id = ?
    #             ''', (current_time, confidence, record[0]))
            
    #         conn.commit()
    #         conn.close()
            
    #         return {"success": True, "message": f"Successfully {action.replace('_', '-')}ed"}
            
    #     except Exception as e:
    #         logger.error(f"Failed to log attendance: {str(e)}")
    #         return {"success": False, "message": str(e)}

    def log_attendance(
        self, 
        employee_id: str, 
        action: str, 
        confidence: float = 0.0,
        latitude: float = None,
        longitude: float = None
) -> Dict:
        """Log attendance (check-in/check-out) with GPS coordinates"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            current_time = datetime.now()
            current_date = current_time.date()
            
            if action == "check_in":
                cursor.execute('''
                    SELECT id FROM attendance_logs 
                    WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL
                ''', (employee_id, current_date))
                
                if cursor.fetchone():
                    conn.close()
                    return {"success": False, "message": "Already checked in today"}
                
                cursor.execute('''
                    INSERT INTO attendance_logs 
                    (employee_id, check_in_time, date, confidence, latitude, longitude)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (employee_id, current_time, current_date, confidence, latitude, longitude))
                
            elif action == "check_out":
                cursor.execute('''
                    SELECT id FROM attendance_logs 
                    WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL AND check_out_time IS NULL
                ''', (employee_id, current_date))
                
                record = cursor.fetchone()
                if not record:
                    conn.close()
                    return {"success": False, "message": "No check-in record found for today"}
                
                cursor.execute('''
                    UPDATE attendance_logs 
                    SET check_out_time = ?, confidence = ?, latitude = ?, longitude = ?
                    WHERE id = ?
                ''', (current_time, confidence, latitude, longitude, record[0]))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": f"Successfully {action.replace('_', '-')}ed"}
            
        except Exception as e:
            logger.error(f"Failed to log attendance: {str(e)}")
            return {"success": False, "message": str(e)}


    def submit_leave_request(self, leave_data: dict) -> dict:
        """Submit a new leave request"""
        try:
            employee_id = leave_data.get("employee_id")
            leave_type = leave_data.get("leave_type")
            start_date = leave_data.get("start_date")
            end_date = leave_data.get("end_date")
            reason = leave_data.get("reason", "")

            # Validate required fields
            if not all([employee_id, leave_type, start_date, end_date]):
                return {"success": False, "message": "Missing required fields"}, 400

            # Validate leave type
            valid_leave_types = ["annual", "sick", "emergency", "personal"]
            if leave_type not in valid_leave_types:
                return {"success": False, "message": "Invalid leave type"}, 400

            # Validate dates
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                if start < datetime.now().date():
                    return {"success": False, "message": "Cannot request leave for past dates"}, 400
                if end < start:
                    return {"success": False, "message": "End date must be after start date"}, 400
            except ValueError:
                return {"success": False, "message": "Invalid date format. Use YYYY-MM-DD"}, 400

            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()

            # Check if employee exists
            cursor.execute('SELECT id FROM employees WHERE employee_id = ?', (employee_id,))
            if not cursor.fetchone():
                conn.close()
                return {"success": False, "message": "Employee not found"}, 404

            # Check for overlapping leave requests
            cursor.execute('''
                SELECT id FROM leave_requests 
                WHERE employee_id = ? AND status = 'approved'
                AND ((start_date <= ? AND end_date >= ?) OR 
                     (start_date <= ? AND end_date >= ?) OR
                     (start_date >= ? AND end_date <= ?))
            ''', (employee_id, end_date, start_date, end_date, start_date, start_date, end_date))
            
            if cursor.fetchone():
                conn.close()
                return {"success": False, "message": "Overlapping leave request exists"}, 400

            # Insert leave request
            cursor.execute('''
                INSERT INTO leave_requests (employee_id, leave_type, start_date, end_date, reason)
                VALUES (?, ?, ?, ?, ?)
            ''', (employee_id, leave_type, start_date, end_date, reason))
            
            request_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return {
                "success": True,
                "message": "Leave request submitted successfully",
                "request_id": request_id
            }, 201

        except Exception as e:
            logger.error(f"Failed to submit leave request: {str(e)}")
            return {"success": False, "message": str(e)}, 500

    def get_leave_status(self, employee_id: str) -> dict:
        """Get leave request history for an employee"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()

            # Check if employee exists
            cursor.execute('SELECT id FROM employees WHERE employee_id = ?', (employee_id,))
            if not cursor.fetchone():
                conn.close()
                return {"success": False, "message": "Employee not found"}, 404

            # Get last 10 leave requests
            cursor.execute('''
                SELECT id, leave_type, start_date, end_date, reason, status, 
                       submitted_at, approved_at, approved_by, rejection_reason
                FROM leave_requests 
                WHERE employee_id = ?
                ORDER BY submitted_at DESC
                LIMIT 10
            ''', (employee_id,))
            
            records = cursor.fetchall()
            conn.close()

            leave_requests = []
            for record in records:
                request = {
                    "request_id": record[0],
                    "leave_type": record[1],
                    "start_date": record[2],
                    "end_date": record[3],
                    "reason": record[4],
                    "status": record[5],
                    "submitted_at": record[6],
                    "approved_at": record[7],
                    "approved_by": record[8],
                    "rejection_reason": record[9]
                }
                leave_requests.append(request)

            return {
                "success": True,
                "leave_requests": leave_requests
            }, 200

        except Exception as e:
            logger.error(f"Failed to get leave status: {str(e)}")
            return {"success": False, "message": str(e)}, 500

    def get_all_leave_requests(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> dict:
        """Get all leave requests for admin view"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()

            query = '''
                SELECT lr.id, lr.employee_id, e.name, lr.leave_type, lr.start_date, lr.end_date, 
                       lr.reason, lr.status, lr.submitted_at, lr.approved_at, lr.approved_by, 
                       lr.rejection_reason
                FROM leave_requests lr
                JOIN employees e ON lr.employee_id = e.employee_id
            '''
            params = []

            if status:
                query += ' WHERE lr.status = ?'
                params.append(status)

            query += ' ORDER BY lr.submitted_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])

            cursor.execute(query, params)
            records = cursor.fetchall()

            # Get total count
            count_query = 'SELECT COUNT(*) FROM leave_requests'
            count_params = []
            if status:
                count_query += ' WHERE status = ?'
                count_params.append(status)
            cursor.execute(count_query, count_params)
            total_count = cursor.fetchone()[0]

            conn.close()

            leave_requests = []
            for record in records:
                start_date = datetime.strptime(record[4], "%Y-%m-%d").date()
                end_date = datetime.strptime(record[5], "%Y-%m-%d").date()
                duration = (end_date - start_date).days + 1
                request = {
                    "request_id": record[0],
                    "employee_id": record[1],
                    "employee_name": record[2],
                    "leave_type": record[3],
                    "start_date": record[4],
                    "end_date": record[5],
                    "duration_days": duration,
                    "reason": record[6],
                    "status": record[7],
                    "submitted_at": record[8],
                    "approved_at": record[9],
                    "approved_by": record[10],
                    "rejection_reason": record[11]
                }
                leave_requests.append(request)

            return {
                "success": True,
                "leave_requests": leave_requests,
                "total_count": total_count,
                "limit": limit,
                "offset": offset
            }, 200

        except Exception as e:
            logger.error(f"Failed to get all leave requests: {str(e)}")
            return {"success": False, "message": str(e)}, 500

    def approve_leave_request(self, request_id: int, approved_by: str) -> dict:
        """Approve a leave request"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()

            # Check if request exists and is pending
            cursor.execute('''
                SELECT status FROM leave_requests WHERE id = ?
            ''', (request_id,))
            record = cursor.fetchone()
            if not record:
                conn.close()
                return {"success": False, "message": "Leave request not found"}, 404
            if record[0] != "pending":
                conn.close()
                return {"success": False, "message": "Leave request is not pending"}, 400

            # Update leave request
            cursor.execute('''
                UPDATE leave_requests 
                SET status = 'approved', approved_at = ?, approved_by = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), approved_by, request_id))

            conn.commit()
            conn.close()

            return {"success": True, "message": "Leave request approved successfully"}, 200

        except Exception as e:
            logger.error(f"Failed to approve leave request: {str(e)}")
            return {"success": False, "message": str(e)}, 500

    def reject_leave_request(self, request_id: int, rejected_by: str, reason: str) -> dict:
        """Reject a leave request"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()

            # Check if request exists and is pending
            cursor.execute('''
                SELECT status FROM leave_requests WHERE id = ?
            ''', (request_id,))
            record = cursor.fetchone()
            if not record:
                conn.close()
                return {"success": False, "message": "Leave request not found"}, 404
            if record[0] != "pending":
                conn.close()
                return {"success": False, "message": "Leave request is not pending"}, 400

            # Update leave request
            cursor.execute('''
                UPDATE leave_requests 
                SET status = 'rejected', rejection_reason = ?, approved_by = ?
                WHERE id = ?
            ''', (reason, rejected_by, request_id))

            conn.commit()
            conn.close()

            return {"success": True, "message": "Leave request rejected successfully"}, 200

        except Exception as e:
            logger.error(f"Failed to reject leave request: {str(e)}")
            return {"success": False, "message": str(e)}, 500

# Initialize the face recognition system
face_system = FaceRecognitionSystem()

def handle_errors(f):
    """Error handling decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"success": False, "message": "Internal server error"}), 500
    return decorated_function

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Face Recognition API is running"})

@app.route('/api/register', methods=['POST'])
@handle_errors
def register_employee_endpoint():
    """Register a new employee via multipart/form-data"""
    employee_id = request.form.get("employee_id")
    name = request.form.get("name")
    department = request.form.get("department", "")
    position = request.form.get("position", "")
    
    image = request.files.get("image")
    
    if not employee_id or not name or not image:
        return jsonify({"success": False, "message": "Missing required fields"}), 400
    
    try:
        pil_image = Image.open(image.stream).convert("RGB")
        image_array = np.array(pil_image)
    except Exception as e:
        return jsonify({"success": False, "message": f"Invalid image: {str(e)}"}), 400
    
    employee_data = {
        "employee_id": employee_id,
        "name": name,
        "department": department,
        "position": position
    }
    
    result = face_system.register_employee(employee_data, image_array)
    
    if result["success"]:
        return jsonify(result), 201
    else:
        return jsonify(result), 400

@app.route('/api/recognize', methods=['POST'])
@handle_errors
def recognize_employee_endpoint():
    """Recognize employee from uploaded image (multipart/form-data)"""
    file = request.files.get("image")

    if not file:
        return jsonify({"success": False, "message": "Image required"}), 400

    try:
        pil_image = Image.open(file.stream).convert("RGB")
        image_array = np.array(pil_image)
    except Exception as e:
        return jsonify({"success": False, "message": f"Invalid image: {str(e)}"}), 400

    result = face_system.recognize_employee(image_array)

    if result["success"]:
        return jsonify(result), 200
    else:
        return jsonify(result), 400

# @app.route('/api/attendance', methods=['POST'])
# @handle_errors
# def attendance():
#     """Combined recognize and attendance logging"""
#     if 'image' not in request.files or 'action' not in request.form:
#         return jsonify({"success": False, "message": "Image and action required"}), 400

#     image_file = request.files['image']
#     action = request.form['action']

#     if action not in ['check_in', 'check_out']:
#         return jsonify({"success": False, "message": "Action must be check_in or check_out"}), 400

#     image_bytes = image_file.read()
#     recognition_result = face_system.recognize_employee(image_bytes)

#     if not recognition_result["success"] or not recognition_result.get("matches"):
#         return jsonify({"success": False, "message": "Employee not recognized"}), 400

#     best_match = max(recognition_result["matches"], key=lambda x: x["confidence"])
#     attendance_result = face_system.log_attendance(
#         best_match["employee_id"], 
#         action,
#         float(best_match["confidence"])
#     )

#     return jsonify({
#         "success": attendance_result["success"],
#         "message": attendance_result["message"],
#         "employee": {
#             "employee_id": best_match["employee_id"],
#             "name": best_match["name"],
#             "confidence": float(best_match["confidence"]),
#             "bbox": best_match.get("bbox")
#         },
#         "action": action,
#         "timestamp": datetime.now().isoformat()
#     })


@app.route('/api/attendance', methods=['POST'])
@handle_errors
def attendance():
    """Combined recognize and attendance logging with GPS"""
    if 'image' not in request.files or 'action' not in request.form:
        return jsonify({"success": False, "message": "Image and action required"}), 400

    image_file = request.files['image']
    action = request.form['action']
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    if action not in ['check_in', 'check_out']:
        return jsonify({"success": False, "message": "Action must be check_in or check_out"}), 400

    # validate coordinates
    if not latitude or not longitude:
        return jsonify({"success": False, "message": "Coordinates required"}), 400
    
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return jsonify({"success": False, "message": "Invalid coordinates"}), 400

    # Recognize employee
    image_bytes = image_file.read()
    recognition_result = face_system.recognize_employee(image_bytes)

    if not recognition_result["success"] or not recognition_result.get("matches"):
        return jsonify({"success": False, "message": "Employee not recognized"}), 400

    best_match = max(recognition_result["matches"], key=lambda x: x["confidence"])

    # Log attendance including GPS
    attendance_result = face_system.log_attendance(
        best_match["employee_id"], 
        action,
        float(best_match["confidence"]),
        latitude=latitude,
        longitude=longitude
    )

    return jsonify({
        "success": attendance_result["success"],
        "message": attendance_result["message"],
        "employee": {
            "employee_id": best_match["employee_id"],
            "name": best_match["name"],
            "confidence": float(best_match["confidence"]),
            "bbox": best_match.get("bbox")
        },
        "action": action,
        "coords": {"lat": latitude, "lng": longitude},
        "timestamp": datetime.now().isoformat()
    })



@app.route('/api/employees', methods=['GET'])
@handle_errors
def get_employees():
    """Get all registered employees"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT employee_id, name, department, position, created_at 
        FROM employees ORDER BY created_at DESC
    ''')
    employees = cursor.fetchall()
    conn.close()
    
    employee_list = []
    for emp in employees:
        employee_list.append({
            "employee_id": emp[0],
            "name": emp[1],
            "department": emp[2],
            "position": emp[3],
            "created_at": emp[4]
        })
    
    return jsonify({"employees": employee_list})

@app.route('/api/attendance/history/<employee_id>', methods=['GET'])
@handle_errors
def get_attendance_history(employee_id):
    """Get attendance history for an employee"""
    days = request.args.get('days', 30, type=int)
    start_date = datetime.now() - timedelta(days=days)
    
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, check_in_time, check_out_time, status, confidence, created_at
        FROM attendance_logs 
        WHERE employee_id = ? AND date >= ?
        ORDER BY date DESC
    ''', (employee_id, start_date.date()))
    
    records = cursor.fetchall()
    conn.close()
    
    history = []
    for record in records:
        history.append({
            "date": record[0],
            "check_in_time": record[1],
            "check_out_time": record[2],
            "status": record[3],
            "confidence": record[4],
            "created_at": record[5]
        })
    
    return jsonify({"attendance_history": history})

@app.route('/api/leave/request', methods=['POST'])
@handle_errors
def submit_leave_request():
    """Submit a new leave request"""
    try:
        leave_data = request.get_json()
        if not leave_data:
            return jsonify({"success": False, "message": "Invalid JSON data"}), 400
        
        result, status_code = face_system.submit_leave_request(leave_data)
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Error in submit_leave_request: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/leave/status/<employee_id>', methods=['GET'])
@handle_errors
def get_leave_status(employee_id):
    """Get leave request history for an employee"""
    result, status_code = face_system.get_leave_status(employee_id)
    return jsonify(result), status_code

@app.route('/api/leave/admin', methods=['GET'])
@handle_errors
def get_all_leave_requests():
    """Get all leave requests for admin view"""
    status = request.args.get('status')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    result, status_code = face_system.get_all_leave_requests(status, limit, offset)
    return jsonify(result), status_code

@app.route('/api/leave/<int:request_id>/approve', methods=['PUT'])
@handle_errors
def approve_leave_request(request_id):
    """Approve a leave request"""
    try:
        data = request.get_json()
        if not data or 'approved_by' not in data:
            return jsonify({"success": False, "message": "Missing approved_by field"}), 400
        
        result, status_code = face_system.approve_leave_request(request_id, data['approved_by'])
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Error in approve_leave_request: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/leave/<int:request_id>/reject', methods=['PUT'])
@handle_errors
def reject_leave_request(request_id):
    """Reject a leave request"""
    try:
        data = request.get_json()
        if not data or 'rejected_by' not in data or 'reason' not in data:
            return jsonify({"success": False, "message": "Missing rejected_by or reason field"}), 400
        
        result, status_code = face_system.reject_leave_request(request_id, data['rejected_by'], data['reason'])
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Error in reject_leave_request: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)