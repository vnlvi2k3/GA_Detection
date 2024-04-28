$('#qty').val('');
	$('#edition').val('4');
	$('#trx_id').val('');
	$('#block_num').val('');
	$('#block_id').val('');
	$('#prev_block_id').val('');
	$('#pot_leg_qty').val('');
	$('#pot_alc_qty').val('');


	function verify() {
		var edition = parseInt($('#edition').val());
		var qty = parseInt($('#qty').val());
		var trx_id = $('#trx_id').val();
		var block_num = parseInt($('#block_num').val());
		var block_id = $('#block_id').val();
		var prev_block_id = $('#prev_block_id').val();

		var packs = generatePacks(qty, md5(trx_id + block_id + prev_block_id), edition, block_num);

		var results = '';
		if (packs.error) {
			results = '<span style="color: red;">' + packs.error + '</span>'
		} else {
			packs.cards.forEach(c => c.details = SM.GetCardDetails(c.card_detail_id))
			packs.cards.sort((a, b) => (b.details.rarity + (b.gold ? 10 : 0)) - (a.details.rarity + (a.gold ? 10 : 0)));

			// build human readable results
			results = [];
			results.push('Cards:');
			results.push('');
			packs.cards.forEach(card => {
				results.push(`${card.uid} : <span class="rarity-${card.details.rarity}">${card.details.name}</span>${card.gold ? ' <span style="color: #E68E16;"><b>(GOLD)</b></span>' : ''}`);
			});
			results = results.join('<br>');
		}

		$('#result, #lnk_explorer').show();
		$('#result').html(results);
		$('#lnk_explorer').attr('href', `https://hiveblocks.com/tx/${trx_id}`);
	}

	function generatePacks(qty, seed, edition, block_num) {
		if(isNaN(edition) || ![0,1,2,4,5].includes(edition))
			return { error: 'Invalid or unspecified pack edition.' };

		console.log('----------------------');
		console.log('Seed: ' + seed);

		var card_list = [];
		var rng = new Math.seedrandom(seed);
		var num_legendary_charges = parseInt($('#pot_leg_qty').val());
		var num_alchemy_charges = parseInt($('#pot_alc_qty').val());

		if(isNaN(num_legendary_charges))
			num_legendary_charges = 0;
		if(isNaN(num_alchemy_charges))
			num_alchemy_charges = 0;

		for(var i = 0; i < qty; i++) {
			let pack_cards = [];
			var rare_or_better = false;

			// Choose 5 unique cards for each pack
			for(var j = 0; j < 5; j++) {
				var legendary_boost = 0, gold_boost = 0;

				if(num_legendary_charges > 0) {
					num_legendary_charges--;
					legendary_boost = 100;
				}

				if(num_alchemy_charges > 0) {
					num_alchemy_charges--;
					gold_boost = 100;
				}

				var card_detail_id = chooseCard((!rare_or_better && j == 5 - 1) ? 2 : 1, rng, edition, block_num, legendary_boost);
				if (card_detail_id === -1) {
					return { error: 'Could not generate cards for the given parameters (block number may be too low for the edition you selected).' };
				}

				if(edition != 2) {
					while(pack_cards.find(c => c.card_detail_id == card_detail_id))
						card_detail_id = chooseCard((!rare_or_better && j == 5 - 1) ? 2 : 1, rng, edition, block_num, legendary_boost);
				}

				// Make sure there is at least one rare or better card per pack
				if(!rare_or_better)
					rare_or_better = SM.cards.find(d => d.id == card_detail_id).rarity > 1;

				if(block_num > 34282507) {
					// Pick a random number of random numbers
					let rng_index = Math.floor(rng() * 15 + 5);
					for(var ri = 0; ri < rng_index; ri++)
						rng();
				}

				// Determine if the card is a gold card
				var gold_rng = rng();
				console.log('Gold RNG: ' + gold_rng + ', Gold Boost: ' + gold_boost);
				var gold = gold_rng < (0.02 * (1 + gold_boost / 100));

				// figure out uid and save final card details
				console.log('Card Detail ID: ' + card_detail_id);
				var prefix = (gold ? 'G' : 'C') + edition + '-' + card_detail_id + '-';
				pack_cards.push({ uid: prefix + generateUid(10, rng), card_detail_id: card_detail_id, gold: gold });
			}

			pack_cards.forEach(card => card_list.push(card));
		}

		console.log('----------------------');
		return { cards: card_list };
	}

	var verifier_card_list = SM.cards.slice().sort((a, b) => a.id - b.id);
	var base_rarity_pcts = [0.248, 0.048, 0.008, 0];
	function chooseCard(min_rarity, rng, edition, block_num, legendary_boost) {
		var rarity_index = rng();
		console.log('Rarity RNG: ' + rarity_index);

		var rarity = 1;
		var rarity_pcts = base_rarity_pcts.slice();

		if(legendary_boost) {
			var boost = +(rarity_pcts[2] * (legendary_boost / 100)).toFixed(3);
			rarity_pcts[0] += boost;
			rarity_pcts[1] += boost;
			rarity_pcts[2] += boost;
		}

		for(var i = 0; i < rarity_pcts.length; i++) {
			rarity = i + 1;

			if(rarity_index > rarity_pcts[i])
				break;
		}

		if(rarity < min_rarity)
			rarity = min_rarity;

		var cards = verifier_card_list.filter(d => d.rarity == rarity && !d.is_promo && d.editions.split(',').map(n => parseInt(n)).includes(edition));

		// Only a subset of the promo edition are available through packs (essence orbs)
		if(edition == 2)
			cards = cards.filter(d => d.id >= 118 && d.id <= 129);

		// Only include cards that were created before the requested block_num
		cards = cards.filter(d => d.created_block_num == null || d.created_block_num <= block_num);

		let card_index = rng();
		console.log(`Card Index: ${card_index}, Cards: ${cards.length}, Legendary Boost: ${legendary_boost}`);
		if(cards.length === 0) {
			return -1;    // indicates no cards were available that match the requested parameters
		}
		return cards[Math.floor(card_index * cards.length)].id;
	}

	function generateUid(length, rng) {
		return Math.round((Math.pow(36, length + 1) - rng() * Math.pow(36, length))).toString(36).slice(1).toUpperCase();
	}
