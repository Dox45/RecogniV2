'use client'
import React, { useState, useEffect,useRef, useCallback } from 'react';
import { Camera, Upload, Users, Calendar, CheckCircle, XCircle, Clock, UserPlus, FileText, Home, Settings, LogOut, User, Building, Briefcase } from 'lucide-react';
import Webcam from 'react-webcam';

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Main App Component
const AttendanceApp = () => {
  const [activeTab, setActiveTab] = useState('employee');
  const [isAdmin, setIsAdmin] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Building className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-xl font-semibold text-gray-900">
                Employee Attendance System
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setIsAdmin(!isAdmin)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isAdmin 
                    ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                    : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                }`}
              >
                {isAdmin ? 'Switch to Employee' : 'Admin Login'}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Tab Navigation */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {!isAdmin ? (
              <>
                <TabButton 
                  active={activeTab === 'employee'} 
                  onClick={() => setActiveTab('employee')}
                  icon={<Clock className="h-4 w-4" />}
                >
                  Attendance
                </TabButton>
                <TabButton 
                  active={activeTab === 'leave'} 
                  onClick={() => setActiveTab('leave')}
                  icon={<Calendar className="h-4 w-4" />}
                >
                  Leave Management
                </TabButton>
              </>
            ) : (
              <>
                <TabButton 
                  active={activeTab === 'register'} 
                  onClick={() => setActiveTab('register')}
                  icon={<UserPlus className="h-4 w-4" />}
                >
                  Register Employee
                </TabButton>
                <TabButton 
                  active={activeTab === 'employees'} 
                  onClick={() => setActiveTab('employees')}
                  icon={<Users className="h-4 w-4" />}
                >
                  Employees
                </TabButton>
                <TabButton 
                  active={activeTab === 'leave-admin'} 
                  onClick={() => setActiveTab('leave-admin')}
                  icon={<FileText className="h-4 w-4" />}
                >
                  Leave Requests
                </TabButton>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!isAdmin ? (
          <>
            {activeTab === 'employee' && <EmployeeAttendance />}
            {activeTab === 'leave' && <LeaveManagement />}
          </>
        ) : (
          <>
            {activeTab === 'register' && <EmployeeRegistration />}
            {activeTab === 'employees' && <EmployeeList />}
            {activeTab === 'leave-admin' && <LeaveAdministration />}
          </>
        )}
      </div>
    </div>
  );
};

// Tab Button Component
const TabButton = ({ children, active, onClick, icon }) => (
  <button
    onClick={onClick}
    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
      active
        ? 'border-blue-500 text-blue-600'
        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
    }`}
  >
    {icon}
    <span>{children}</span>
  </button>
);

// Employee Attendance Component
const videoConstraints = {
  facingMode: 'user',
  width: { ideal: 480 }, // Use ideal to avoid OverconstrainedError
  height: { ideal: 360 }
};

const EmployeeAttendance = () => {
  const webcamRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [isCameraReady, setIsCameraReady] = useState(false);

  // Check camera permissions on mount
  useEffect(() => {
    const checkPermissions = async () => {
      try {
        const permissionStatus = await navigator.permissions.query({ name: 'camera' });
        if (permissionStatus.state === 'denied') {
          setCameraError('Camera permission denied. Please allow camera access in your browser settings.');
        } else if (permissionStatus.state === 'prompt') {
          // Permission not yet granted; will be requested when Webcam initializes
          setCameraError(null);
        } else {
          setCameraError(null);
        }
      } catch (err) {
        console.error('Error checking permissions:', err);
        setCameraError('Unable to check camera permissions.');
      }
    };
    checkPermissions();
  }, []);

  // Capture a screenshot from webcam
  const capture = useCallback(() => {
    if (webcamRef.current && isCameraReady) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setCapturedImage(imageSrc);
        setResult(null);
      } else {
        setCameraError('Failed to capture image. Please ensure the camera is working.');
      }
    } else {
      setCameraError('Camera is not ready. Please wait or check camera settings.');
    }
  }, [webcamRef, isCameraReady]);

  // Send image to API
  const handleAttendance = async (action) => {
    if (!capturedImage) {
      setResult({ success: false, message: 'Please capture a photo first', action });
      return;
    }

    setLoading(true);
    try {
      const blob = await (await fetch(capturedImage)).blob();
      const formData = new FormData();
      formData.append('image', blob, 'capture.jpg');
      formData.append('action', action);

      const response = await fetch(`${API_BASE_URL}/attendance`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult({ ...data, action });
    } catch (error) {
      console.error('Attendance API error:', error);
      setResult({
        success: false,
        message: 'Failed to connect to server',
        action
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle webcam errors
  const handleUserMediaError = (error) => {
    console.error('Webcam error:', error);
    let errorMessage = 'Failed to access camera: ';
    if (error.name === 'NotAllowedError') {
      errorMessage += 'Permission denied. Please allow camera access.';
    } else if (error.name === 'OverconstrainedError') {
      errorMessage += 'Camera constraints not supported. Try a different device or resolution.';
    } else if (error.name === 'NotFoundError') {
      errorMessage += 'No camera found. Please connect a camera.';
    } else {
      errorMessage += error.message;
    }
    setCameraError(errorMessage);
    setIsCameraReady(false);
  };

  // Handle webcam ready
  const handleUserMedia = () => {
    setIsCameraReady(true);
    setCameraError(null);
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Employee Attendance</h2>
        <p className="text-gray-600">Use your camera to check in or check out</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        {cameraError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
            <div className="flex items-center space-x-2">
              <XCircle className="h-5 w-5 text-red-600" />
              <span className="font-medium text-red-800">Camera Error</span>
            </div>
            <p className="mt-1 text-red-700">{cameraError}</p>
          </div>
        )}

        {!capturedImage ? (
          <div className="flex flex-col items-center">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="rounded-lg shadow"
              onUserMediaError={handleUserMediaError}
              onUserMedia={handleUserMedia}
            />
            <button
              onClick={capture}
              disabled={!isCameraReady}
              className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-md font-medium flex items-center space-x-2 disabled:opacity-50"
            >
              <Camera className="h-5 w-5" />
              <span>Capture</span>
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <img src={capturedImage} alt="Captured" className="rounded-lg shadow max-h-64" />
            <div className="mt-4 flex space-x-4">
              <button
                onClick={() => handleAttendance('check_in')}
                disabled={loading}
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-md font-medium disabled:opacity-50 flex items-center space-x-2"
              >
                <CheckCircle className="h-4 w-4" />
                <span>{loading ? 'Processing...' : 'Check In'}</span>
              </button>
              <button
                onClick={() => handleAttendance('check_out')}
                disabled={loading}
                className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-md font-medium disabled:opacity-50 flex items-center space-x-2"
              >
                <XCircle className="h-4 w-4" />
                <span>{loading ? 'Processing...' : 'Check Out'}</span>
              </button>
              <button
                onClick={() => setCapturedImage(null)}
                className="bg-gray-300 hover:bg-gray-400 text-gray-800 px-6 py-3 rounded-md font-medium"
              >
                Retake
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Result Display */}
      {result && (
        <div className={`rounded-lg p-4 ${
          result.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
        }`}>
          <div className="flex items-center space-x-2">
            {result.success ? (
              <CheckCircle className="h-5 w-5 text-green-600" />
            ) : (
              <XCircle className="h-5 w-5 text-red-600" />
            )}
            <span className={`font-medium ${result.success ? 'text-green-800' : 'text-red-800'}`}>
              {result.success ? 'Success!' : 'Error'}
            </span>
          </div>
          <p className={`mt-1 ${result.success ? 'text-green-700' : 'text-red-700'}`}>
            {result.message}
          </p>
        </div>
      )}
    </div>
  );
};


// Leave Management Component
const LeaveManagement = () => {
  const [activeSection, setActiveSection] = useState('request');
  const [leaveRequest, setLeaveRequest] = useState({
    employeeId: '',
    startDate: '',
    endDate: '',
    reason: '',
    type: 'annual'
  });
  const [checkId, setCheckId] = useState('');
  const [leaveStatus, setLeaveStatus] = useState(null);
  const [submissionResult, setSubmissionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch leave status when checkId changes
  useEffect(() => {
    if (checkId) {
      const fetchLeaveStatus = async () => {
        setLoading(true);
        setError(null);
        try {
          const response = await fetch(`${API_BASE_URL}/leave/status/${checkId}`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
          });
          const data = await response.json();
          if (data.success) {
            setLeaveStatus(data.leave_requests);
          } else {
            setError(data.message);
          }
        } catch (err) {
          setError('Failed to fetch leave status');
          console.error('Error fetching leave status:', err);
        } finally {
          setLoading(false);
        }
      };
      fetchLeaveStatus();
    }
  }, [checkId]);

  // Handle input changes for leave request form
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setLeaveRequest((prev) => ({ ...prev, [name]: value }));
  };

  // Submit leave request to API
  const handleLeaveRequest = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSubmissionResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/leave/request`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          employee_id: leaveRequest.employeeId,
          leave_type: leaveRequest.type,
          start_date: leaveRequest.startDate,
          end_date: leaveRequest.endDate,
          reason: leaveRequest.reason
        })
      });
      const data = await response.json();
      setSubmissionResult({
        success: data.success,
        message: data.message,
        requestId: data.request_id
      });

      if (data.success) {
        // Reset form
        setLeaveRequest({
          employeeId: '',
          startDate: '',
          endDate: '',
          reason: '',
          type: 'annual'
        });
      }
    } catch (err) {
      setError('Failed to submit leave request');
      console.error('Error submitting leave request:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Leave Management</h2>
        <p className="text-gray-600">Request a leave or check your leave status</p>
      </div>

      {/* Tabs for switching between request and status check */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={() => setActiveSection('request')}
          className={`px-4 py-2 font-medium rounded-md ${
            activeSection === 'request' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'
          }`}
        >
          Request Leave
        </button>
        <button
          onClick={() => setActiveSection('status')}
          className={`px-4 py-2 font-medium rounded-md ${
            activeSection === 'status' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'
          }`}
        >
          Check Status
        </button>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        {activeSection === 'request' ? (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Submit Leave Request</h3>
            <form onSubmit={handleLeaveRequest} className="space-y-4">
              <div>
                <label htmlFor="employeeId" className="block text-sm font-medium text-gray-700">
                  Employee ID
                </label>
                <input
                  type="text"
                  id="employeeId"
                  name="employeeId"
                  value={leaveRequest.employeeId}
                  onChange={handleInputChange}
                  className="mt-1 block w-full text-black border border-gray-300 rounded-md p-2"
                  required
                />
              </div>
              <div>
                <label htmlFor="type" className="block text-sm font-medium text-gray-700">
                  Leave Type
                </label>
                <select
                  id="type"
                  name="type"
                  value={leaveRequest.type}
                  onChange={handleInputChange}
                  className="mt-1 block w-full text-black border border-gray-300 rounded-md p-2"
                  required
                >
                  <option value="annual">Annual</option>
                  <option value="sick">Sick</option>
                  <option value="emergency">Emergency</option>
                  <option value="personal">Personal</option>
                </select>
              </div>
              <div>
                <label htmlFor="startDate" className="block text-sm font-medium text-gray-700">
                  Start Date
                </label>
                <input
                  type="date"
                  id="startDate"
                  name="startDate"
                  value={leaveRequest.startDate}
                  onChange={handleInputChange}
                  className="mt-1 block w-full text-black border border-gray-300 rounded-md p-2"
                  required
                />
              </div>
              <div>
                <label htmlFor="endDate" className="block text-sm font-medium text-gray-700">
                  End Date
                </label>
                <input
                  type="date"
                  id="endDate"
                  name="endDate"
                  value={leaveRequest.endDate}
                  onChange={handleInputChange}
                  className="mt-1 block w-full text-black border border-gray-300 rounded-md p-2"
                  required
                />
              </div>
              <div>
                <label htmlFor="reason" className="block text-sm font-medium text-gray-700">
                  Reason
                </label>
                <textarea
                  id="reason"
                  name="reason"
                  value={leaveRequest.reason}
                  onChange={handleInputChange}
                  className="mt-1 block w-full text-black border border-gray-300 rounded-md p-2"
                  rows="4"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-md font-medium disabled:opacity-50 flex items-center space-x-2"
              >
                <span>{loading ? 'Submitting...' : 'Submit Request'}</span>
              </button>
            </form>

            {/* Submission Result */}
            {submissionResult && (
              <div
                className={`mt-4 rounded-lg p-4 ${
                  submissionResult.success
                    ? 'bg-green-50 border border-green-200'
                    : 'bg-red-50 border border-red-200'
                }`}
              >
                <div className="flex items-center space-x-2">
                  {submissionResult.success ? (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-600" />
                  )}
                  <span
                    className={`font-medium ${
                      submissionResult.success ? 'text-green-800' : 'text-red-800'
                    }`}
                  >
                    {submissionResult.success ? 'Success!' : 'Error'}
                  </span>
                </div>
                <p
                  className={`mt-1 ${
                    submissionResult.success ? 'text-green-700' : 'text-red-700'
                  }`}
                >
                  {submissionResult.message}
                  {submissionResult.requestId && (
                    <span> Request ID: {submissionResult.requestId}</span>
                  )}
                </p>
              </div>
            )}
          </div>
        ) : (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Check Leave Status</h3>
            <div className="space-y-4">
              <div>
                <label htmlFor="checkId" className="block text-sm font-medium text-gray-700">
                  Employee ID
                </label>
                <input
                  type="text"
                  id="checkId"
                  value={checkId}
                  onChange={(e) => setCheckId(e.target.value)}
                  className="mt-1 block w-full border border-gray-300 rounded-md p-2"
                />
              </div>

              {loading && <p className="text-gray-600">Loading...</p>}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2">
                    <XCircle className="h-5 w-5 text-red-600" />
                    <span className="font-medium text-red-800">Error</span>
                  </div>
                  <p className="mt-1 text-red-700">{error}</p>
                </div>
              )}
              {leaveStatus && (
                <div className="mt-4">
                  <h4 className="text-md font-medium text-gray-900">Leave Requests</h4>
                  {leaveStatus.length === 0 ? (
                    <p className="text-gray-600">No leave requests found.</p>
                  ) : (
                    <ul className="mt-2 space-y-2">
                      {leaveStatus.map((request) => (
                        <li
                          key={request.request_id}
                          className="border border-gray-200 rounded-md p-3"
                        >
                          <p>
                            <strong>Type:</strong> {request.leave_type}
                          </p>
                          <p>
                            <strong>Start Date:</strong> {request.start_date}
                          </p>
                          <p>
                            <strong>End Date:</strong> {request.end_date}
                          </p>
                          <p>
                            <strong>Status:</strong> {request.status}
                          </p>
                          {request.reason && (
                            <p>
                              <strong>Reason:</strong> {request.reason}
                            </p>
                          )}
                          {request.rejection_reason && (
                            <p>
                              <strong>Rejection Reason:</strong> {request.rejection_reason}
                            </p>
                          )}
                          <p>
                            <strong>Submitted:</strong> {new Date(request.submitted_at).toLocaleString()}
                          </p>
                          {request.approved_at && (
                            <p>
                              <strong>Approved:</strong> {new Date(request.approved_at).toLocaleString()}
                            </p>
                          )}
                          {request.approved_by && (
                            <p>
                              <strong>Approved By:</strong> {request.approved_by}
                            </p>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Employee Registration Component
const EmployeeRegistration = () => {
  const [formData, setFormData] = useState({
    employee_id: '',
    name: '',
    department: '',
    position: ''
  });
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) {
      alert('Please select an image');
      return;
    }

    setLoading(true);
    const formDataToSend = new FormData();
    Object.keys(formData).forEach(key => {
      formDataToSend.append(key, formData[key]);
    });
    formDataToSend.append('image', selectedImage);

    try {
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        body: formDataToSend,
      });
      const data = await response.json();
      setResult(data);
      if (data.success) {
        setFormData({
          employee_id: '',
          name: '',
          department: '',
          position: ''
        });
        setSelectedImage(null);
        setPreview('');
      }
    } catch (error) {
      setResult({
        success: false,
        message: 'Failed to connect to server'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Register New Employee</h2>
        <p className="text-gray-600">Add employee details and photo for face recognition</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 max-w-2xl mx-auto">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-black mb-1">
                Employee ID *
              </label>
              <input
                type="text"
                name="employee_id"
                value={formData.employee_id}
                onChange={handleInputChange}
                className="w-full px-3 py-2 text-black border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Full Name *
              </label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                className="w-full px-3 py-2 text-black border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Department
              </label>
              <input
                type="text"
                name="department"
                value={formData.department}
                onChange={handleInputChange}
                className="w-full px-3 py-2 text-black border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Position
              </label>
              <input
                type="text"
                name="position"
                value={formData.position}
                onChange={handleInputChange}
                className="w-full px-3 py-2 text-black border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Image Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Employee Photo *
            </label>
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  {preview ? (
                    <img src={preview} alt="Preview" className="max-h-48 max-w-full object-contain rounded" />
                  ) : (
                    <>
                      <Upload className="w-10 h-10 mb-3 text-gray-400" />
                      <p className="mb-2 text-sm text-gray-500">
                        <span className="font-semibold">Click to upload</span> employee photo
                      </p>
                      <p className="text-xs text-gray-500">PNG, JPG, JPEG (MAX. 10MB)</p>
                    </>
                  )}
                </div>
                <input
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageSelect}
                  required
                />
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md font-medium disabled:opacity-50 flex items-center justify-center space-x-2"
          >
            <UserPlus className="h-4 w-4" />
            <span>{loading ? 'Registering...' : 'Register Employee'}</span>
          </button>
        </form>

        {result && (
          <div className={`mt-4 rounded-lg p-4 ${
            result.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}>
            <div className="flex items-center space-x-2">
              {result.success ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <XCircle className="h-5 w-5 text-red-600" />
              )}
              <span className={`font-medium ${result.success ? 'text-green-800' : 'text-red-800'}`}>
                {result.success ? 'Success!' : 'Error'}
              </span>
            </div>
            <p className={`mt-1 ${result.success ? 'text-green-700' : 'text-red-700'}`}>
              {result.message}
            </p>
            {result.success && result.employee_id && (
              <p className="mt-1 text-sm text-green-600">
                Employee ID: {result.employee_id}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Employee List Component
const EmployeeList = () => {
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [attendanceHistory, setAttendanceHistory] = useState([]);

  React.useEffect(() => {
    fetchEmployees();
  }, []);

  const fetchEmployees = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/employees`);
      const data = await response.json();
      setEmployees(data.employees || []);
    } catch (error) {
      console.error('Failed to fetch employees:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchAttendanceHistory = async (employeeId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/attendance/history/${employeeId}`);
      const data = await response.json();
      setAttendanceHistory(data.attendance_history || []);
    } catch (error) {
      console.error('Failed to fetch attendance history:', error);
      setAttendanceHistory([]);
    }
  };

  const handleViewHistory = (employee) => {
    setSelectedEmployee(employee);
    fetchAttendanceHistory(employee.employee_id);
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading employees...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Employee Management</h2>
        <p className="text-gray-600">View all registered employees and their attendance history</p>
      </div>

      {!selectedEmployee ? (
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b">
            <h3 className="text-lg font-medium text-gray-900">
              All Employees ({employees.length})
            </h3>
          </div>
          
          {employees.length === 0 ? (
            <div className="text-center py-12">
              <Users className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">No employees registered yet</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Employee
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Department
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Position
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Registered
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {employees.map((employee) => (
                    <tr key={employee.employee_id}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center">
                            <User className="h-5 w-5 text-gray-600" />
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">
                              {employee.name}
                            </div>
                            <div className="text-sm text-gray-500">
                              {employee.employee_id}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {employee.department || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {employee.position || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(employee.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button
                          onClick={() => handleViewHistory(employee)}
                          className="text-blue-600 hover:text-blue-900"
                        >
                          View History
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSelectedEmployee(null)}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              ← Back to Employee List
            </button>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center space-x-4 mb-6">
              <div className="h-16 w-16 rounded-full bg-gray-300 flex items-center justify-center">
                <User className="h-8 w-8 text-gray-600" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">{selectedEmployee.name}</h3>
                <p className="text-gray-600">{selectedEmployee.employee_id}</p>
                <p className="text-sm text-gray-500">
                  {selectedEmployee.department} • {selectedEmployee.position}
                </p>
              </div>
            </div>

            <div className="border-t pt-6">
              <h4 className="text-lg font-medium text-gray-900 mb-4">Attendance History</h4>
              
              {attendanceHistory.length === 0 ? (
                <div className="text-center py-8">
                  <Calendar className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No attendance records found</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Check In
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Check Out
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Confidence
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {attendanceHistory.map((record, index) => (
                        <tr key={index}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {new Date(record.date).toLocaleDateString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {record.check_in_time ? 
                              new Date(record.check_in_time).toLocaleTimeString() : '-'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {record.check_out_time ? 
                              new Date(record.check_out_time).toLocaleTimeString() : '-'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              record.status === 'present' ? 'bg-green-100 text-green-800' : 
                              'bg-red-100 text-red-800'
                            }`}>
                              {record.status}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {record.confidence ? `${(record.confidence * 100).toFixed(1)}%` : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Leave Administration Component
const LeaveAdministration = () => {
  const [leaveRequests, setLeaveRequests] = useState([]);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [actionResult, setActionResult] = useState(null);

  // Fetch leave requests on mount and when filterStatus changes
  useEffect(() => {
    const fetchLeaveRequests = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `${API_BASE_URL}/leave/admin${filterStatus !== 'all' ? `?status=${filterStatus}` : ''}`,
          {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
          }
        );
        const data = await response.json();
        if (data.success) {
          setLeaveRequests(data.leave_requests);
        } else {
          setError(data.message);
        }
      } catch (err) {
        setError('Failed to fetch leave requests');
        console.error('Error fetching leave requests:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchLeaveRequests();
  }, [filterStatus]);

  // Handle approve action
  const handleApprove = async (requestId) => {
    setLoading(true);
    setError(null);
    setActionResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/leave/${requestId}/approve`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved_by: 'admin' }) // Replace 'admin' with actual admin ID or name
      });
      const data = await response.json();
      setActionResult({
        success: data.success,
        message: data.message
      });

      if (data.success) {
        setLeaveRequests((prev) =>
          prev.map((req) =>
            req.request_id === requestId ? { ...req, status: 'approved', approved_at: new Date().toISOString(), approved_by: 'admin' } : req
          )
        );
        if (selectedRequest && selectedRequest.request_id === requestId) {
          setSelectedRequest((prev) => ({
            ...prev,
            status: 'approved',
            approved_at: new Date().toISOString(),
            approved_by: 'admin'
          }));
        }
      }
    } catch (err) {
      setError('Failed to approve leave request');
      console.error('Error approving leave request:', err);
    } finally {
      setLoading(false);
    }
  };

  // Handle reject action
  const handleReject = async (requestId) => {
    setLoading(true);
    setError(null);
    setActionResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/leave/${requestId}/reject`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rejected_by: 'admin', reason: 'Request denied by admin' }) // Replace with actual admin ID and dynamic reason
      });
      const data = await response.json();
      setActionResult({
        success: data.success,
        message: data.message
      });

      if (data.success) {
        setLeaveRequests((prev) =>
          prev.map((req) =>
            req.request_id === requestId
              ? { ...req, status: 'rejected', rejection_reason: 'Request denied by admin', approved_by: 'admin' }
              : req
          )
        );
        if (selectedRequest && selectedRequest.request_id === requestId) {
          setSelectedRequest((prev) => ({
            ...prev,
            status: 'rejected',
            rejection_reason: 'Request denied by admin',
            approved_by: 'admin'
          }));
        }
      }
    } catch (err) {
      setError('Failed to reject leave request');
      console.error('Error rejecting leave request:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredRequests = leaveRequests.filter(
    (req) => filterStatus === 'all' || req.status === filterStatus
  );

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved':
        return 'bg-green-100 text-green-800';
      case 'rejected':
        return 'bg-red-100 text-red-800';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getLeaveTypeColor = (type) => {
    switch (type) {
      case 'annual':
        return 'bg-blue-100 text-blue-800';
      case 'sick':
        return 'bg-purple-100 text-purple-800';
      case 'emergency':
        return 'bg-red-100 text-red-800';
      case 'personal':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Leave Administration</h2>
        <p className="text-gray-600">Manage employee leave requests</p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
          <div className="flex items-center space-x-2">
            <XCircle className="h-5 w-5 text-red-600" />
            <span className="font-medium text-red-800">Error</span>
          </div>
          <p className="mt-1 text-red-700">{error}</p>
        </div>
      )}

      {/* Action Result Display */}
      {actionResult && (
        <div
          className={`rounded-lg p-4 mb-4 ${
            actionResult.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}
        >
          <div className="flex items-center space-x-2">
            {actionResult.success ? (
              <CheckCircle className="h-5 w-5 text-green-600" />
            ) : (
              <XCircle className="h-5 w-5 text-red-600" />
            )}
            <span className={`font-medium ${actionResult.success ? 'text-green-800' : 'text-red-800'}`}>
              {actionResult.success ? 'Success!' : 'Error'}
            </span>
          </div>
          <p className={`mt-1 ${actionResult.success ? 'text-green-700' : 'text-red-700'}`}>
            {actionResult.message}
          </p>
        </div>
      )}

      {!selectedRequest ? (
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">
                Leave Requests ({filteredRequests.length})
              </h3>
              <div className="flex items-center space-x-4">
                <label className="text-sm font-medium text-gray-700">Filter:</label>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All</option>
                  <option value="pending">Pending</option>
                  <option value="approved">Approved</option>
                  <option value="rejected">Rejected</option>
                </select>
              </div>
            </div>
          </div>

          {loading && <p className="text-center py-12 text-gray-600">Loading...</p>}
          {!loading && filteredRequests.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">No leave requests found</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Employee
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Submitted
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredRequests.map((request) => (
                    <tr key={request.request_id}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {request.employee_name}
                          </div>
                          <div className="text-sm text-gray-500">
                            {request.employee_id}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getLeaveTypeColor(request.leave_type)}`}>
                          {request.leave_type}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {new Date(request.start_date).toLocaleDateString()} -{' '}
                        {new Date(request.end_date).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(request.status)}`}>
                          {request.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(request.submitted_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                        <button
                          onClick={() => setSelectedRequest(request)}
                          className="text-blue-600 hover:text-blue-900"
                        >
                          View Details
                        </button>
                        {request.status === 'pending' && (
                          <>
                            <button
                              onClick={() => handleApprove(request.request_id)}
                              className="text-green-600 hover:text-green-900"
                              disabled={loading}
                            >
                              Approve
                            </button>
                            <button
                              onClick={() => handleReject(request.request_id)}
                              className="text-red-600 hover:text-red-900"
                              disabled={loading}
                            >
                              Reject
                            </button>
                          </>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSelectedRequest(null)}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              ← Back to Leave Requests
            </button>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-gray-900">Leave Request Details</h3>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(selectedRequest.status)}`}>
                {selectedRequest.status.toUpperCase()}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Employee</label>
                  <p className="text-sm text-gray-900">{selectedRequest.employee_name}</p>
                  <p className="text-sm text-gray-500">{selectedRequest.employee_id}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Leave Type</label>
                  <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getLeaveTypeColor(selectedRequest.leave_type)}`}>
                    {selectedRequest.leave_type}
                  </span>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Start Date</label>
                  <p className="text-sm text-gray-900">{new Date(selectedRequest.start_date).toLocaleDateString()}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">End Date</label>
                  <p className="text-sm text-gray-900">{new Date(selectedRequest.end_date).toLocaleDateString()}</p>
                </div>
              </div>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Submitted At</label>
                  <p className="text-sm text-gray-900">{new Date(selectedRequest.submitted_at).toLocaleString()}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Duration</label>
                  <p className="text-sm text-gray-900">{selectedRequest.duration_days} days</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Reason</label>
                  <p className="text-sm text-gray-900">{selectedRequest.reason || 'N/A'}</p>
                </div>
                {selectedRequest.rejection_reason && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Rejection Reason</label>
                    <p className="text-sm text-gray-900">{selectedRequest.rejection_reason}</p>
                  </div>
                )}
                {selectedRequest.approved_by && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Approved By</label>
                    <p className="text-sm text-gray-900">{selectedRequest.approved_by}</p>
                  </div>
                )}
              </div>
            </div>

            {selectedRequest.status === 'pending' && (
              <div className="mt-8 flex justify-end space-x-4">
                <button
                  onClick={() => handleReject(selectedRequest.request_id)}
                  className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md font-medium flex items-center space-x-2"
                  disabled={loading}
                >
                  <XCircle className="h-4 w-4" />
                  <span>Reject</span>
                </button>
                <button
                  onClick={() => handleApprove(selectedRequest.request_id)}
                  className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md font-medium flex items-center space-x-2"
                  disabled={loading}
                >
                  <CheckCircle className="h-4 w-4" />
                  <span>Approve</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};


export default AttendanceApp;