import React from 'react';
import './LiveBar.css';

const LiveBar = ({ realPercentage, fakePercentage }) => {
  // Default to 50% if undefined.
  const real = realPercentage !== undefined ? realPercentage : 50;
  const fake = fakePercentage !== undefined ? fakePercentage : 50;

  return (
    <div className="live-bar-container">
      <div className="live-bar-real" style={{ width: `${real}%` }}>
        <span className="live-bar-label">{real.toFixed(2)}%</span>
      </div>
      <div className="live-bar-fake" style={{ width: `${fake}%` }}>
        <span className="live-bar-label">{fake.toFixed(2)}%</span>
      </div>
    </div>
  );
};

export default LiveBar;
