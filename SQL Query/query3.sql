select store.storename as "Nama Toko", sum(transaction.qty ) as "Jumlah Quantity"
from store
inner join transaction
on store.storeid=transaction.storeid
group by "Nama Toko"
order by "Jumlah Quantity" desc
