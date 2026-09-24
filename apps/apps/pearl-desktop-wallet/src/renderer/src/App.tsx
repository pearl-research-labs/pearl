import {HashRouter as Router, Routes, Route, useLocation} from 'react-router-dom';
import WelcomePage from './pages/WelcomePage';
import ImportAccount from './pages/ImportAccount';
import CreateWallet from './pages/create-wallet/CreateWallet';
import WalletDashboard from './pages/WalletDashboard';
import ActivityPage from './pages/ActivityPage';
import SendTransaction from './pages/send-transaction/SendTransaction';
import ReceiveTransaction from './pages/ReceiveTransaction';
import WalletUnlock from './pages/WalletUnlock';
import ChangePassword from './pages/ChangePassword';
import {useNavigate} from 'react-router-dom';
import {SyncWallet} from './SyncWallet';
import {MajorUpgradeBanner} from './components/MajorUpgradeBanner';
import './App.css';

function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <div className="relative flex h-screen w-full flex-col overflow-hidden bg-gradient-to-br from-gray-50 dark:from-background to-gray-100 dark:to-card font-sans text-gray-900 dark:text-foreground antialiased">
      <MajorUpgradeBanner />
      <div className={`min-h-0 flex-1`}>
        <Routes>
          <Route path="/" element={<WelcomePage />} />
          <Route path="/wallet" element={<WalletDashboard />} />
          <Route path="/send" element={<SendTransaction />} />
          <Route path="/receive" element={<ReceiveTransaction />} />
          <Route path="/unlock" element={<WalletUnlock />} />
          <Route path="/change-password" element={<ChangePassword />} />
          <Route path="/import-account" element={<ImportAccount />} />
          <Route path="/onboarding/create" element={<CreateWallet />} />
          <Route path="/activity" element={<ActivityPage onBack={() => navigate('/wallet')} />} />
          <Route
            path="*"
            element={
              <div style={{color: 'red', padding: '20px'}}>
                Route not found: {location.pathname}
              </div>
            }
          />
        </Routes>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <SyncWallet />
      <AppContent />
    </Router>
  );
}

export default App;
