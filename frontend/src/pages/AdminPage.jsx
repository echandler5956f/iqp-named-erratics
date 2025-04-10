import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Layout/Header';
import axios from 'axios';
import './AdminPage.css';

function AdminPage() {
  const [erratics, setErratics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [userInfo, setUserInfo] = useState(null);
  const navigate = useNavigate();

  // Authentication check
  useEffect(() => {
    const token = localStorage.getItem('token');
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    
    if (!token || !user.is_admin) {
      navigate('/login');
      return;
    }
    
    setUserInfo(user);
    
    // Set up axios auth header
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    
    // Load erratics data
    fetchErratics();
  }, [navigate]);

  // Fetch all erratics
  const fetchErratics = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:3001/api/erratics');
      setErratics(response.data);
    } catch (err) {
      console.error('Failed to fetch erratics:', err);
      setError('Failed to load erratic data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  if (loading) {
    return (
      <div className="admin-page">
        <Header />
        <div className="admin-container">
          <div className="loading-message">Loading data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="admin-page">
      <Header />
      <div className="admin-container">
        <div className="admin-header">
          <h1>Admin Dashboard</h1>
          <div className="admin-controls">
            <div className="user-info">
              Logged in as: <strong>{userInfo?.username}</strong>
            </div>
            <button className="logout-button" onClick={handleLogout}>Logout</button>
          </div>
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        <div className="admin-actions">
          <button className="action-button">Add New Erratic</button>
          <button className="action-button">Import Data</button>
        </div>
        
        <div className="erratic-list-container">
          <h2>Manage Erratics ({erratics.length})</h2>
          
          <div className="erratic-list">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Name</th>
                  <th>Rock Type</th>
                  <th>Size (m)</th>
                  <th>Location</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {erratics.map(erratic => (
                  <tr key={erratic.id}>
                    <td>{erratic.id}</td>
                    <td>{erratic.name}</td>
                    <td>{erratic.rock_type || 'Unknown'}</td>
                    <td>{erratic.size_meters || 'Unknown'}</td>
                    <td>
                      {erratic.location.coordinates[1].toFixed(5)}, 
                      {erratic.location.coordinates[0].toFixed(5)}
                    </td>
                    <td className="action-cell">
                      <button className="table-action-btn">Edit</button>
                      <button className="table-action-btn delete">Delete</button>
                    </td>
                  </tr>
                ))}
                
                {erratics.length === 0 && (
                  <tr>
                    <td colSpan="6" className="no-data">No erratics found. Add one to get started.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdminPage; 