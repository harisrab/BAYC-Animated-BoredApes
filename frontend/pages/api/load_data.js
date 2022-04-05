const fs = require('fs')
const papa = require('papaparse')

export default function handler(req, res) {
  const file = fs.createReadStream('./public/apes_availability.csv')
  var count = 0 // cache the running count

  papa
    .parse(file, {
      worker: true, // Don't bog down the main thread if its a big file
      header: true,
      // step: function (result) {
      //   // do stuff with result
      // },
      complete: function (results, file) {
        console.log('parsing complete read', count, 'records.')

        console.log("Results Data ====> ", results.data)

        var myApes = JSON.stringify(results.data);
        res.status(200).json(myApes)
      },
    })
}
