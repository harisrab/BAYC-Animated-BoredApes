import * as React from 'react'
import { DataGrid } from '@mui/x-data-grid'

function getID(params) {
  return `${params.row.id}`
}

const columns = [
    {
      field: 'id',
      headerName: 'ID',
      width: 100,
      valueGetter: getID,
    },
  { field: 'apecode', headerName: 'Ape Code', width: 300 },
  { field: 'landmarked', headerName: 'Landmarked?', width: 130 },
]

const rows = [
  { id: 0, landmarked: 'Yes', apecode: 'sdflkajdlkjfasf' },
  { id: 1, landmarked: 'Yes', apecode: 'Cdsfkjjjslafersei' },
  { id: 2, landmarked: 'Yes', apecode: 'asdlfjalkdj' },
  { id: 3, landmarked: 'Yes', apecode: 'adsafadfdfasdfsaf' },
  { id: 4, landmarked: 'Yes', apecode: 'adssdfasdfadf' },
]

export default function SingleRowSelectionGrid() {
  return (
    <div style={{ height: 400, width: '100%' }}>
      <DataGrid rows={rows} columns={columns} />
    </div>
  )
}
