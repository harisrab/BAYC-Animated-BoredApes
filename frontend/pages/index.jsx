import React, { useState, useEffect } from 'react'
import { render } from 'react-dom'
import { Bubble } from 'react-chartjs-2'
import Chart from 'chart.js/auto'
import 'chartjs-plugin-dragdata'
import axios from 'axios'
import { StyledEngineProvider } from '@mui/material/styles'
import SingleRowSelectionGrid from './components/demo'

const data = {
  labels: ['January'],
  datasets: [
    {
      //   label: 'My First dataset',
      //   fill: false,
      lineTension: 0.2,
      backgroundColor: 'rgba(75,192,192,0.4)',
      borderColor: 'rgba(75,192,192,1)',
      //  borderCapStyle: 'butt',
      borderDash: [],
      borderDashOffset: 0.0,
      borderJoinStyle: 'miter',
      pointBorderColor: 'rgba(204,0,0,1)',
      pointBackgroundColor: '#fff',
      pointBorderWidth: 2,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: 'rgba(75,192,192,1)',
      pointHoverBorderColor: 'rgba(220,220,220,1)',
      pointHoverBorderWidth: 2,
      pointRadius: 0.5,
      pointHitRadius: 2,
      data: [
        { y: 0, x: 0, r: 3 },
        { y: 5, x: 200, r: 3 },
        { y: 5, x: 50, r: 3 },
        { y: 100, x: 100, r: 3 },
        { y: 5, x: 20, r: 3 },
        { y: 10, x: 15, r: 3 },
      ],
    },
  ],
}

const options = {
  layout: {
    padding: {
      left: -30,
    },
  },
  scales: {
    x: { min: 0, max: 256, ticks: { display: false } },
    y: { min: 0, max: 256, ticks: { display: false } },
  },
  aspectRatio: 1,
  plugins: {
    dragData: { dragX: true, dragY: true },
    legend: {
      display: false,
    },
  },
}

const App = () => {
  const [currentImg, setCurrentImg] = useState(
    'https://lh3.googleusercontent.com/b7OuAKbk2H9tTp5YZeWRUueVQw6gi9_CTF6ZeG45d2SnczejUdyO6KhVb04IMYdHU5A6kpJuGaWU-NN84suTpL2wqQ1b6LjqQyD9mw=w600'
  )

  useEffect(() => {
    axios
      .get('/api/load_data')
      .then((res) => {
        console.log(res)
      })
      .catch((err) => {
        console.log(err)
      })
  }, [])

  return (
    <div className="w-[1000px] h-screen flex items-center justify-center">
      <StyledEngineProvider injectFirst>
        <SingleRowSelectionGrid />
      </StyledEngineProvider>

      <div
        className="w-[400px] h-[400px]"
        style={{
          backgroundImage: `url(${currentImg})`,
          backgroundRepeat: 'no-repeat',
          backgroundSize: 'cover',
        }}
      >
        <Bubble data={data} options={options} />
      </div>
    </div>
  )
}

export default App
