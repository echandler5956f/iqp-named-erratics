import Header from '../components/Layout/Header';
import LoginForm from '../components/Auth/LoginForm';
import './LoginPage.css';

function LoginPage() {
  return (
    <div className="login-page">
      <Header />
      <LoginForm />
    </div>
  );
}

export default LoginPage; 