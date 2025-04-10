import ErraticsMap from '../components/Map/ErraticsMap';
import Header from '../components/Layout/Header';
import './HomePage.css';

function HomePage() {
  return (
    <div className="home-page">
      <Header />
      <ErraticsMap />
    </div>
  );
}

export default HomePage; 